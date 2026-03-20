
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告知名字无需其他内容"
)

# chain = prompt | model | model
# response = chain.invoke({"lastname":"张", "gender":"女儿"})
# 报错：ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.

# 使用 StrOutputParser 转换第1个model的输出AIMessage，StrOutputParser表示AIMessage转为字符串后，作为第2个model的输入字符串
strOutputParser = StrOutputParser()
chain = prompt | model | strOutputParser | model
response = chain.invoke({"lastname":"张", "gender":"女儿"})

print(type(response)) # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content) # 你好！你提到“张若曦”，。。。。。。

# 方式2：不使用 response.content打印输出，而使用 StrOutputParser 做类型转换
print("\n========== 方式2：不使用 response.content打印输出，而使用 StrOutputParser 做类型转换 ")
chain2 = prompt | model | strOutputParser | model | strOutputParser
response2 = chain2.invoke({"lastname":"张", "gender":"女儿"})
print(type(response2)) # <class 'langchain_core.messages.base.TextAccessor'>
print(response2) # 你好！你提到“张若溪”。。。。。。