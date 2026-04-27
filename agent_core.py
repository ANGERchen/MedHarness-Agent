import json
import threading
from openai import OpenAI
from tools_registry import TOOLS_REGISTRY, TOOLS_SCHEMA
from rag_engine import rag_engine
from memory_manager import memory

main_client = OpenAI(api_key="sk-local", base_url="http://localhost:8000/v1")
vl_client = OpenAI(api_key="sk-local", base_url="http://localhost:8001/v1")

# ================= 优化点：场景意图路由 =================
def intent_router(query: str) -> str:
    prompt = f"仅回答场景标签(medical_qa, image_analysis, daily_tool)之一。输入: {query}"
    res = main_client.chat.completions.create(model="qwen-14b-main", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content.strip()

# ================= 优化点：安全护栏 =================
def guardrail_check(draft: str, context: str) -> str:
    prompt = f"你是医疗安全质检员。检查以下草稿是否包含致命错误或脱离参考资料。草稿：{draft}\n资料：{context}\n若安全则输出原草稿，若危险则输出修正版。"
    return main_client.chat.completions.create(model="qwen-14b-main", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

# ================= 优化点：旁路记忆反思 =================
def async_memory_reflection(user_id, chat_history):
    def _reflect():
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
        prompt = f"从对话中提取用户的永久健康特征（如过敏、慢性病），以纯 JSON 返回如 {{\"allergy\": \"海鲜\"}}，无则返回空字典 {{}}:\n{history_str}"
        try:
            res = main_client.chat.completions.create(model="qwen-14b-main", messages=[{"role": "user", "content": prompt}])
            tags = json.loads(res.choices[0].message.content)
            if tags: memory.update_long_term_tags(user_id, tags)
        except: pass
    threading.Thread(target=_reflect).start()

# ================= 主控流 =================
def run_health_agent(user_message, user_id, image_path=None, chat_history=None):
    profile = memory.get_profile(user_id)
    steps, sleep = memory.get_latest_metrics(user_id)
    intent = intent_router(user_message)
    print(f"🚦 路由意图命中: {intent}")
    
    # 动态构建工具集
    active_tools = []
    rag_context = ""
    if intent == "daily_tool":
        active_tools = [t for t in TOOLS_SCHEMA if t["function"]["name"] in ["get_weather", "plan_exercise"]]
    if intent == "medical_qa":
        rag_context = rag_engine.search(user_message, main_client)
        active_tools = [t for t in TOOLS_SCHEMA if t["function"]["name"] == "web_search"]
    elif intent == "image_analysis" and image_path:
        active_tools = [t for t in TOOLS_SCHEMA if t["function"]["name"] == "analyze_nutrition"]
    else:
        active_tools = [t for t in TOOLS_SCHEMA if t["function"]["name"] == "get_weather"]

    sys_prompt = f"你是专业健康管家。用户画像:{profile.get('tags', {})}。当前状态：今日步数{steps}，昨晚睡眠{sleep}h。本地资料:{rag_context}。请结合资料作答。"
    messages = [{"role": "system", "content": sys_prompt}]
    if chat_history: messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    # 一阶段：工具调度
    response = main_client.chat.completions.create(
        model="qwen-14b-main", messages=messages, tools=active_tools if active_tools else None
    )
    
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            f_name = tc.function.name
            f_args = json.loads(tc.function.arguments)
            if f_name == "analyze_nutrition": f_args.update({"image_path": image_path, "vl_client": vl_client})
            if f_name == "plan_exercise": f_args["user_id"] = user_id
            res_content = TOOLS_REGISTRY[f_name](**f_args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": f_name, "content": res_content})
            
        final_res = main_client.chat.completions.create(model="qwen-14b-main", messages=messages)
        draft = final_res.choices[0].message.content
    else:
        draft = msg.content

    # 二阶段：护栏拦截与持久化
    safe_response = guardrail_check(draft, rag_context)
    memory.log_chat(user_id, user_message, rag_context, safe_response)
    async_memory_reflection(user_id, chat_history)
    
    return safe_response