# langchain通用提示词模版
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

# 提示词模版类PromptTemplate，是Runnable接口的实现类，它可以加入到langchain中的链条
# zero-shot 零样本学习
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}， 刚生了{gender}, 你帮我起个名字，简单回答。"
)

model = Tongyi(model="qwen-max")

# 创建链对象
chain = prompt_template | model
result = chain.invoke(input={"lastname":"张", "gender":"女儿"})
print(result)
# 张婉儿