
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import pickle
import logging

logger=logging.getLogger(__name__)

class retrieval:
    def __init__(self,model,key):
        self.embedding_model = model
        self.dense_vectorstore = None
        self.dense_retrieval = None
        self.sparse_retriever = None
        self.topk=key

    def _rrf_retrival(self,query,k=5,parent_docs=None):
        '''
            通过RRF算法检索
        '''
        logger.info("FAISS稠密检索...")
        self.dense_vectorstore = FAISS.load_local(
            folder_path="dense_vectorstore",
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        self.dense_retrieval=self.dense_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        dense_results = self.dense_retrieval.invoke(query)

        logger.info("BM25稀疏检索...")
        with open("sparse_vectorstore.pkl", "rb") as f:
            self.sparse_retrieval = pickle.load(f)
        sparse_results = self.sparse_retrieval.invoke(query)

        logger.info("rrf合并结果中...")
        # RRF算法融合结果
        k=60
        total_score={}
        for i, doc in enumerate(dense_results):
            total_score[doc.metadata.get("parent_id")] = total_score.get(doc.metadata.get("parent_id"), 0) + 1/(i+k)
        for i, doc in enumerate(sparse_results):
            total_score[doc.metadata.get("parent_id")] = total_score.get(doc.metadata.get("parent_id"), 0) + 1/(i+k)
        sorted_results_id=sorted(total_score.keys(), key=lambda x: total_score[x], reverse=True)
        final_results=[]
        for doc_id in sorted_results_id:
            for doc in parent_docs:
                if doc.metadata.get("parent_id")==doc_id:
                    final_results.append(doc)
        return final_results[0]

