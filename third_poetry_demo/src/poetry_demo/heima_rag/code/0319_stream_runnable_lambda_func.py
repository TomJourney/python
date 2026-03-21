from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

str_parser = StrOutputParser()

model = ChatTongyi(model="qwen3-max")

first_template = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告诉我名字，不需要额外信息"
)

second_template = PromptTemplate.from_template(
    "姓名{name}， 请帮我解析含义"
)

# 使用RunnableLambda类创建自定义函数
# 函数的入参： AIMessage -> dict({"name":"xxx"})
my_func = RunnableLambda(lambda ai_msg : {"name":ai_msg.content})

# 基于RunnableLambda函数构建langchain链
chain = first_template | model | my_func | second_template | model | str_parser
# 流式输出调用llm
result = chain.stream({"lastname":"张", "gender":"女儿"})

for chunk in result:
    print(chunk, end="", flush=True) # 当然可以！我们来解析一下“张婉清”这个名字的含义。......

# 方式2 ： 直接把RunnableLambda自定义函数加入链
print("\n\n ========== 方式2： 直接把RunnableLambda自定义函数加入链: ")
chain2 = (first_template | model | RunnableLambda(lambda ai_msg : {"name":ai_msg.content})
          | second_template | model | str_parser)
result2 = chain2.invoke({"lastname":"张", "gender":"女儿"}) #
print(result2)