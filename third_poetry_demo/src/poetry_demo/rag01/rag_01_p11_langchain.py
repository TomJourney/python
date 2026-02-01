# 1. 加载文档
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
)
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

#  创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_split_result = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embed_model)
vector_store.add_documents(doc_split_result)

# 5. 构建用户查询
question = "黑悟空有哪些游戏场景？"

# 6. 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
joined_retrieved_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
print("joined_retrieved_docs = ", joined_retrieved_docs)

# 7. 构建提示模版
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
                    基于以下上下文，回答问题。如果上下文中没有相关信息，
                    请说"我无法从提供的上下文中找到相关信息"。
                    上下文: {context}
                    问题: {question}
                    回答:"""
)

# 8. 使用大模型生成回答
from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

answer = llm.invoke(prompt.format(question=question, context=joined_retrieved_docs))
print(answer)
