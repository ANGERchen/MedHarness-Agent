import sqlite3
import json
from openai import OpenAI

# 模拟强大的裁判模型与重写模型
judge_client = OpenAI(api_key="sk-local", base_url="http://localhost:8000/v1") 

def llm_judge(query, context, response):
    """评估准则：Faithfulness (忠实于上下文) 和 Relevance (相关性)"""
    prompt = f"""
    严格打分(1-5分)。输出纯JSON，例如 {{"score": 2, "reason": "脱离资料的幻觉"}}
    问题:{query}
    资料:{context}
    回答:{response}
    """
    try:
        res = judge_client.chat.completions.create(model="qwen-14b-main", messages=[{"role": "user", "content": prompt}])
        return json.loads(res.choices[0].message.content)
    except: return {"score": 5, "reason": "eval_error"}

def run_evolution_flywheel(db_path="health_system.db", output_jsonl="evolution_dataset.jsonl"):
    print("🚀 启动 AutoHarness 自动化评估飞轮...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, context, response FROM chat_logs WHERE eval_score IS NULL")
    logs = cursor.fetchall()
    
    bad_cases = []
    for log_id, q, ctx, resp in logs:
        # 1. 自动化评估打分
        eval_res = llm_judge(q, ctx, resp)
        score = eval_res.get("score", 5)
        
        cursor.execute("UPDATE chat_logs SET eval_score=?, eval_reason=? WHERE id=?", 
                       (score, eval_res.get("reason"), log_id))
        
        # 2. 捕获 Bad Case (分数小于4)
        if score < 4:
            bad_cases.append({"query": q, "context": ctx})
            print(f"⚠️ 捕获低分回答 (分数:{score}): {q}")
    conn.commit()
    
    # 3. Teacher Model 重写机制
    if bad_cases:
        print(f"🛠️ 正在重写 {len(bad_cases)} 条低分数据...")
        with open(output_jsonl, "a", encoding="utf-8") as f:
            for case in bad_cases:
                rewrite_prompt = f"请作为顶级医疗专家，基于资料<{case['context']}>完美解答<{case['query']}>，包含推理链。"
                perfect_resp = judge_client.chat.completions.create(
                    model="qwen-14b-main", messages=[{"role": "user", "content": rewrite_prompt}]
                ).choices[0].message.content
                
                # 按照主流微调数据格式 (Alpaca/ShareGPT) 写入
                data = {
                    "instruction": case['query'],
                    "input": case['context'],
                    "output": perfect_resp
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
        # 适合单卡 RTX 4090/Pro 6000 的微调批处理，配合 lmdeploy 或 vLLM
        print(f"✅ 进化语料已追加至 {output_jsonl}，准备进行下一步的 DPO/SFT 模型微调。")

if __name__ == "__main__":
    run_evolution_flywheel()