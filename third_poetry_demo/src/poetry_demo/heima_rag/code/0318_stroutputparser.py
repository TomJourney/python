
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告知名字无需其他内容"
)

chain = prompt | model | model
response = chain.invoke({"lastname":"张", "gender":"女儿"})
# 报错：ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.

print(response.content)

