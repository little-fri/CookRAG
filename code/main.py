import os
import sys
from pathlib import Path
import logging
sys.path.append(os.path.dirname(__file__))
from ragModule.llm_generation import llm_generation
from ragModule.retrieval import retrieval
from ragModule.prepare import PrepareDocuments
from ragModule.index import Index

data_path="../data"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]：%(name)s-%(message)s"
)
logger=logging.getLogger(__name__)
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

def __main__():
    # 初始化
    logger.info("开始数据预处理")
    prepare = PrepareDocuments(data_path)
    prepare.load_documents()
    prepare.chunk_documents()
    logger.info("结束数据预处理")

    logger.info("开始创建索引")
    indexer = Index(model_name="BAAI/bge-small-zh-v1.5")
    
    indexer.vectorstore_setup(prepare.chunk_doc)

    
    llm=llm_generation()
    llm.llm_setup()
    retriever = retrieval(model=indexer.embedding_model,key=5)
    while True:
        query = input("请输入查询（输入 'exit' 退出）：")
        if query.lower() == 'exit':
            break
        
        rewriter_query=llm.rewriter(query)
        route=llm.router(rewriter_query)
        context=retriever._rrf_retrival(rewriter_query,k=5,parent_docs=prepare.parent_doc)
        answer=llm.answer(rewriter_query,context,route)
        print(f"回答: {answer}")

if __name__ == "__main__":
    __main__()
        
        