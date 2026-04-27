import base64
import requests
from duckduckgo_search import DDGS
from memory_manager import memory

def get_weather(location: str):
    try: return requests.get(f"https://wttr.in/{location}?format=%l:+%c+%t", timeout=5).text
    except: return "天气数据暂不可用"

def nutrition_agent(image_path: str, query: str, vl_client):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    res = vl_client.chat.completions.create(
        model="qwen2-vl",
        messages=[{"role": "user", "content": [{"type":"text","text":f"营养师：{query}"},
                  {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}]}]
    )
    return res.choices[0].message.content

def web_search_agent(query: str):
    try:
        results = DDGS().text(query, max_results=3)
        context = "\n".join([f"-[全网资讯] 标题:{r['title']} | 摘要:{r['body']}" for r in results])
        return f"全网实时检索：\n{context}" if context else "网络未命中。"
    except Exception as e:
        return f"搜索异常: {str(e)}"
def exercise_planner(user_id, status_query):
    steps, sleep = memory.get_latest_metrics(user_id)
    return f"【系统同步】今日步数:{steps}, 昨晚睡眠:{sleep}h。请基于此数据给用户建议：{status_query}"


TOOLS_REGISTRY = {
    "get_weather": get_weather,
    "analyze_nutrition": nutrition_agent,
    "web_search": web_search_agent,
    "plan_exercise": exercise_planner
}

# 对应的 Schema，用于 LLM 识别
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_weather", "description": "查天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "analyze_nutrition", "description": "识别食物营养", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "全网搜索最新医疗资讯", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "plan_exercise", "description": "结合用户步数和睡眠制定运动计划", "parameters": {"type": "object", "properties": {"status_query": {"type": "string"}}, "required": ["status_query"]}}}
]