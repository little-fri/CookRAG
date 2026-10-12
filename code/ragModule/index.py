from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import pickle
import logging

logger=logging.getLogger(__name__)
class Index:
    def __init__(self,model_name):
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_model_setup()
        
        self.dense_vectorstore = None
        self.sparse_vectorstore = None

    def embedding_model_setup(self):
        '''
            启动embedding模型
        '''
        logger.info("启动embedding模型")
        emb_model=HuggingFaceEmbeddings(
            model_name=self.model_name, 
            model_kwargs={"device": "cuda"}
        )
        self.embedding_model = emb_model

    def vectorstore_setup(self,chunk_docs):
        '''
            启动向量存储
        '''
        logger.info("开始进行FAISS向量存储")
        dense_vectorstore = FAISS.from_documents(
            documents=chunk_docs,
            embedding=self.embedding_model
        )
        FAISS.save_local(dense_vectorstore, "dense_vectorstore")

        logger.info("开始进行bm25向量存储")
        sparse_vectorstore = BM25Retriever.from_documents(
            documents=chunk_docs,
            k=5
        )
        with open("sparse_vectorstore.pkl", "wb") as f:
            pickle.dump(sparse_vectorstore, f)
