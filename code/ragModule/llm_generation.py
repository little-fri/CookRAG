from langchain_openai import ChatOpenAI
import os
import logging

logger=logging.getLogger(__name__)
class llm_generation:
    def __init__(self, model_name="deepseek-flash",max_tokens=8192,temperature=0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = None

    def llm_setup(self):
        '''
            启动llm模型
        '''
        logger.info("启动llm模型")
        if os.getenv("DeepSeekAPIKey")is None:
            logger.critical("DeepSeekAPIKey 没在环境中")
        else:
            llm = ChatOpenAI(
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                api_key=os.getenv("DeepSeekAPIKey"),
                base_url="https://api.deepseek.com",
            )
            self.llm = llm

    def rewriter(self,query):
        '''
            重写查询
        '''
        logger.info("llm正在重写query...")
        prompt=f"请将以下查询改写为更清晰、更具体的形式，以便更好地检索相关信息：\n\n{query}\n\n请提供改写后的查询。"
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def router(self,query):
        '''
            路由查询
        '''
        logger.info("llm正在判断query类别...")
        prompt=f"""请根据以下查询，判断它最属于哪一个主题类别[list,detail,general]：

        用户提问：
        {query}

        注意仅回答主题类别的名称，不要提供任何解释或额外信息。
        以下是主题类别的定义：
        1. list: 查询涉及列出多个项目、选项或实体的请求。比如用户提问：请给我推荐好吃的菜，此时只回答菜的名称即可
        2. detail: 查询涉及获取某个特定项目、选项或实体的详细信息。比如用户提问：告诉我西红柿炒鸡蛋做法，此时回答菜的详细做法
        3. general: 查询涉及一般性的信息、概念或背景知识。比如用户提问：告诉我哪些菜是中国菜，此时回答中国菜的概念和背景知识
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def answer(self,query,context,route):
        '''
            回答查询
        '''
        logger.info("llm正在输出答案...")
        if route=="list":
            prompt=f"请根据以下上下文\n\n{context}\n\n和提问\n\n{query}\n\n，提供一个简洁的列表形式的回答：\n\n{query}\n\n请确保回答内容清晰、完整，并尽量涵盖相关信息。比如用户提问：请给我推荐好吃的菜，此时只回答菜的名称即可"
        elif route=="detail":
            prompt=f"请根据以下上下文\n\n{context}\n\n和提问\n\n{query}\n\n，提供详细且准确的回答：\n\n{query}\n\n请确保回答内容清晰、完整，并尽量涵盖相关信息。"
        else:
            prompt=f"请根据以下上下文\n\n{context}\n\n和提问\n\n{query}\n\n，提供一个准确的回答：\n\n{query}\n\n请确保回答内容清晰、完整，并尽量涵盖相关信息。"
        response = self.llm.invoke(prompt)
        return response.content.strip()


