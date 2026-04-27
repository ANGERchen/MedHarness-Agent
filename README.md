# 🧬 MedHarness-Agent

> **工业级多模态自进化医疗 AI 智能体架构**
> 
> 本项目不仅是一个简单的 AI 助手，它构建了一套完整的 **LMM-Ops (Large Multimodal Model Operations)** 闭环：融合了动态场景路由、高级 RAG 管道（Query Rewrite + Rerank）以及基于 Pairwise 盲测引擎的 **AutoHarness 自进化测试台**。

## 🌟 核心架构亮点

### 1. 智能意图路由 (Intelligent Orchestration)
摒弃了全量加载工具的冗余模式，系统根据用户输入动态路由至特定场景（医疗问诊、视觉分析、健康规划），有效降低了 Token 损耗并显著提升了复杂指令的遵循率。

### 2. 深度优化的 RAG 引擎
针对医学领域口语化提问，集成了：
- **Query Rewrite**: 将患者主诉转化为标准医学检索实体。
- **BGE Reranker**: 引入交叉编码器对初筛文档进行重排序，确保输出建议的权威性与精准度。

### 3. 自进化评测台 (Self-Evolution Harness)
这是本项目的核心壁垒。系统通过旁路埋点自动收集交互日志，并触发：
- **Pairwise Judge**: 利用强模型作为裁判，对 Agent 与 Baseline 进行盲测对比。
- **DPO 数据飞轮**: 自动捕获低分 Bad Cases，由 Teacher Model 重写后生成标准 DPO 格式数据集，驱动模型持续进化。

## 🛠️ 技术栈
- **核心模型**: Qwen2.5-14B / Qwen2-VL (多模态)
- **推理框架**: vLLM / lmdeploy (OpenAI-compatible API)
- **向量数据库**: ChromaDB
- **算法增强**: Sentence-Transformers + Cross-Encoder
- **交互界面**: Streamlit

## 🚀 快速启动

### 1. 环境准备
```bash
pip install -r requirements.txt

### 2. 启动智能体应用
streamlit run web_app/app.py
### 3. 开启自进化飞轮
# 自动评估历史对话并生成 DPO 训练数据集
python harness/auto_harness.py
