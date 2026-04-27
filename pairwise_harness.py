import json
import sqlite3
from openai import OpenAI

# 假设我们在本地使用 vLLM 或 lmdeploy 拉起了多个端点
# 或者用不同的 model_name 区分
judge_client = OpenAI(api_key="sk-local", base_url="http://localhost:8000/v1")

class DPOHarness:
    def __init__(self, db_path="health_system.db"):
        self.conn = sqlite3.connect(db_path)

    def pairwise_judge(self, query: str, context: str, answer_a: str, answer_b: str) -> dict:
        """核心优化：盲测对比引擎"""
        prompt = f"""
        你是一个严苛的医疗AI评估专家。请比较两个模型对同一问题的回答。
        
        [诊断场景与用户输入]: {query}
        [病理活检/金标准参考]: {context}
        
        [模型 A 回答]: {answer_a}
        [模型 B 回答]: {answer_b}
        
        评估准则：
        1. 忠实度：绝不允许出现参考资料之外的医学幻觉。
        2. 专业性：在处理相似易混淆疾病时，能否准确提炼差异特征。
        3. 简洁性：直接给出结论，拒绝废话。
        
        请输出纯 JSON 格式，决定哪个更好（"A", "B", 或 "Tie"），并给出原因。
        示例: {{"winner": "A", "reason": "A精准指出了特征，而B存在事实错误"}}
        """
        try:
            res = judge_client.chat.completions.create(
                model="qwen-max", # 使用最强模型做裁判
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            return {"winner": "Tie", "reason": "评估异常"}

    def run_dpo_pipeline(self, output_file="dpo_dataset.jsonl"):
        print("🚀 启动 Pairwise 对比测试台与 DPO 数据飞轮...")
        
        # 构造一个高难度的医学测试例 (Hard Case)
        # 比如：具有相似影像学特征的易混淆疾病
        test_case = {
            "query": "患者下颌骨无痛性肿胀，影像学呈现磨玻璃样改变。请根据资料给出诊断建议。",
            "context": "病理活检显示：基质呈高度纤维化，成骨细胞缺乏，骨小梁呈C型或V型不规则排列。符合骨纤维异常增殖症（FD）特征，需与骨化性纤维瘤（OF）进行鉴别，后者通常具有明显的包膜和成骨细胞环绕。",
            "agent_response": "考虑为骨化性纤维瘤（OF），建议立刻进行广泛切除。", # 这是一个错误的幻觉回答
            "baseline_response": "根据病理活检缺乏成骨细胞且骨小梁呈C/V型，诊断倾向于骨纤维异常增殖症（FD）。与OF的鉴别要点在于OF通常有成骨细胞环绕及明显包膜。" # 这是一个完美回答
        }
        
        # 1. 执行盲测 (打乱顺序以防止位置偏见)
        print("⚖️ 正在进行裁判模型盲测对比...")
        eval_result = self.pairwise_judge(
            query=test_case["query"], 
            context=test_case["context"], 
            answer_a=test_case["agent_response"], 
            answer_b=test_case["baseline_response"]
        )
        
        # 2. 映射胜利者，构造 DPO 数据对
        winner = eval_result.get("winner")
        print(f"🏆 评测结果: 模型 {winner} 获胜。原因: {eval_result.get('reason')}")
        
        if winner in ["A", "B"]:
            chosen = test_case["agent_response"] if winner == "A" else test_case["baseline_response"]
            rejected = test_case["baseline_response"] if winner == "A" else test_case["agent_response"]
            
            # DPO 标准格式：包含 prompt, chosen (好回答), rejected (坏回答)
            dpo_data = {
                "prompt": f"请结合资料回答：\n资料：{test_case['context']}\n问题：{test_case['query']}",
                "chosen": chosen,
                "rejected": rejected
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo_data, ensure_ascii=False) + "\n")
                
            print(f"✅ DPO 偏好对已写入 {output_file}，准备进入 RLHF 阶段。")

if __name__ == "__main__":
    harness = DPOHarness()
    harness.run_dpo_pipeline()