from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

"""
PromptTemplate ->(extends) StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

# 测试format
template = PromptTemplate.from_template("我的邻居是 {lastname}，最喜欢{hobby}")
result = template.format(lastname="张三", hobby="钓鱼")
print(result) # 我的邻居是 张三，最喜欢钓鱼
print(type(result)) # <class 'str'>

# 测试invoke
result2 = template.invoke({"lastname":"李四", "hobby":"唱歌"})
print(result2) # text='我的邻居是 李四，最喜欢唱歌'
print(type(result2)) # <class 'langchain_core.prompt_values.StringPromptValue'>

