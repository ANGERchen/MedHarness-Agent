import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class AdvancedRAG:
    def __init__(self, db_path="./medical_knowledge_db"):
        print("正在加载 Embedder 和 Reranker 模型...")
        self.embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        # 引入 BGE 重排模型提升精度
        self.reranker = CrossEncoder('BAAI/bge-reranker-base') 
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="medical_docs")

    def search(self, query: str, llm_client, top_k=2):
        # 1. 查询改写 (Query Rewrite)
        rewrite_prompt = f"将以下口语化的患者问题转化为专业的医学检索关键词（用空格隔开，提取最核心的实体）：{query}"
        try:
            rewrite_res = llm_client.chat.completions.create(
                model="qwen-14b-main", 
                messages=[{"role": "user", "content": rewrite_prompt}]
            )
            search_query = rewrite_res.choices[0].message.content.strip()
        except:
            search_query = query
            
        print(f"🔄 内部 RAG 改写：{query} -> {search_query}")

        # 2. 扩大范围多路召回
        emb = self.embed_model.encode(search_query).tolist()
        results = self.collection.query(query_embeddings=[emb], n_results=top_k * 3)
        docs = results['documents'][0] if results['documents'] else []
        
        if not docs: return "本地知识库未找到权威参考。"

        # 3. 交叉编码器重排序 (Rerank)
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # 将文档和得分绑定并倒序排列
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked_docs[:top_k]]
        
        return "\n".join([f"参考资料[{i+1}]: {text}" for i, text in enumerate(top_docs)])

rag_engine = AdvancedRAG()