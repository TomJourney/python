# 用React实现简单agent
import os
from langchain_classic import hub

promp = hub.pull("hwchas17/react")
print(promp)

# 导入deepseek
from langchain_deepseek import ChatDeepSeek

# 选择要使用的大模型
llm = ChatDeepSeek(model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY"))

# 导入 SerpAPIWrapper 即工具包
from langchain_community.utilities import SerpAPIWrapper

