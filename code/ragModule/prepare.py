import uuid
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

class PrepareDocuments:
    '''
        用于预处理文档
    '''
    def __init__(self,data_path) -> List[Document]: 
        # 初始化:文档来源，父文档，子块，以及他们的映射关系
        self.data_path = Path(data_path)
        self.parent_doc: List[Document] = []
        self.chunk_doc: List[Document] = []
        self.parent_chunk_map={}

    def load_documents(self):
        '''
            加载所有md文件
        '''
        total_doc=[]
        for md_file in self.data_path.rglob("*md"):
            with open(md_file,'r',encoding='utf-8') as f:
                content=f.read()
            id=uuid.uuid4()
            doc=Document(
                page_content=content,
                metadata={
                    "parent_id":str(id),
                    "source":str(md_file)
                }
            )
            total_doc.append(doc)
        self.parent_doc=total_doc

    def chunk_documents(self):
        '''
            将文档切分成块
        '''
        total_chunk=[]
        headers_to_split_on=[
            ("#","Header 1"),
            ("##","Header 2"),
            ("###","Header 3"),
        ]
        text_spliter=MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        for current_doc in self.parent_doc:
            chunks=text_spliter.split_text(current_doc.page_content)
            for chunk in chunks:
                id=uuid.uuid4()
                chunk_doc=Document(
                    page_content=chunk.page_content,
                    metadata={
                        "parent_id":current_doc.metadata["parent_id"],
                        "chunk_id":str(id),
                        "source":current_doc.metadata["source"]
                    }
                )
                self._enhance_metadata(chunk_doc)
                total_chunk.append(chunk_doc)
        self.chunk_doc=total_chunk

    def _enhance_metadata(self,chunk_doc:Document):
        '''
            增强metadata信息
        '''
        # TODO: 可以在这里添加更多的metadata信息，比如chunk的长度，创建时间等
        pass

a=PrepareDocuments(data_path="../../data")
a.load_documents()
a.chunk_documents()


