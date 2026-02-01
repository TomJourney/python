from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.deepseek import DeepSeek
import os
from pathlib import Path
from dotenv import load_dotenv


# ################ note: 因兼容性问题， 程序运行不成功

# 加载环境变量
load_dotenv()

# 创建deepseek模型
deekSeekLLM = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 2 加载数据，形成文档
file_path = os.path.join(Path.home(), "studynote", "workbench", "python", "poetry-demo", "data", "heishenhua01.txt")
documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

# 3 基于文档构建索引
index = VectorStoreIndex.from_documents(documents)

# 4 基于索引创建问答引擎
query_engine = index.as_query_engine(llm=deekSeekLLM)

# 5 基于引擎开始问答
print(query_engine.query("<黑神话：悟空>中有哪些战斗工具"))

