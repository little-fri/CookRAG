# CookRAG
<<<<<<< HEAD
### 一.使用
**1）创建虚拟环境与对应的环境**
> `conda create -n cook-rag python=3.12`
> `pip install -r ./code/requirements.txt`

**2）配置DeepSeekAPIKey**
> `export DeepSeekAPIKey=xxx`

**3）使用**
> `python3 ./code/main.py`

### 二.文件夹结构
/CookRAG
|——code（代码）
│   ├── dense_vectorstore（FAISS向量存储）
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── main.py（主程序）
│   ├── ragModule（RAG模块）
│   │   ├── index.py（索引构建）
│   │   ├── llm_generation.py（llm生成）
│   │   ├── prepare.py（数据预处理）
│   │   └── retrieval.py（检索模块）
│   ├── requirements.txt（环境）
│   └── sparse_vectorstore.pkl（BM25向量存储）
├── data（Cook数据源）

=======
参考All-In-RAG（https://datawhalechina.github.io/all-in-rag/）的实践项目写的简单RAG系统。通过RRF混合检索实现菜谱查询
>>>>>>> origin/main
