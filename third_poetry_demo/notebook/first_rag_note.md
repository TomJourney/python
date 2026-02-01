# 【README】

1. 本文内容代码总结自《大模型应用开发-rag实战课》，作者黄佳；
   1. 作者源代码地址： [https://github.com/huangjia2019/rag-in-action/tree/master/00-%E7%AE%80%E5%8D%95RAG-SimpleRAG](https://github.com/huangjia2019/rag-in-action/tree/master/00-%E7%AE%80%E5%8D%95RAG-SimpleRAG)
2. 本文记录了第1个rag应用，该应用基于langchain，deepseek实现。
3. 本文python应用使用poetry管理依赖包；
4. python版本是3.11.9 ；
   1. <font color=red>注意：python版本不能太高，否则可能导致兼容性问题;</font> (经测试，python3.14.3无法与sentence-transformers依赖包最新版本兼容)

---

# 【1】第1个rag应用

1. 源码参见： [https://github.com/TomJourney/python/tree/main/third_poetry_demo](https://github.com/TomJourney/python/tree/main/third_poetry_demo)

2. 本文所有依赖包由poetry管理，依赖包定义在 third_poetry_demo/pyproject.toml文件中。 安装好poetry后，执行运行 poetry install 即可（根据pyproject.toml中定义的依赖包进行安装）

## 【1.1】rag应用具体逻辑

1. 背景：根据黑神话悟空的wiki介绍，基于rag实现关于黑神话悟空的QA系统；

【 [rag_01_p11_langchain.py](../src/poetry_demo/rag01/rag_01_p11_langchain.py) 】

```python
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

```

---

## 【1.2】deepseek响应效果

```json
content='根据提供的上下文，黑神话：悟空的游戏场景融合了中国的文化和自然地标。具体提到的取景地包括：\n\n*   重庆的大足石刻\n*   山西省的小西天、南禅寺、铁佛寺、广胜寺和鹳雀楼\n\n此外，上下文还提及游戏共有36个取景地，其中27个位于山西省。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 75, 'prompt_tokens': 1437, 'total_tokens': 1512, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 1408}, 'prompt_cache_hit_tokens': 1408, 'prompt_cache_miss_tokens': 29}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '663aa899-2708-4b65-9555-b52ae654e691', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c197c-bd06-7793-aaba-c3b0bc042141-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 1437, 'output_tokens': 75, 'total_tokens': 1512, 'input_token_details': {'cache_read': 1408}, 'output_token_details': {}}
```

