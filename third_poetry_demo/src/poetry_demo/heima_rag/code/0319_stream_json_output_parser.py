from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

first_template = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，并封装到JSON格式返回给我，"
    "要求key是name，value是起的名字。请严格遵守格式要求"
)

second_template = PromptTemplate.from_template(
    "姓名{name}， 请帮我解析含义"
)

# 构建langchain链
chain = first_template | model | json_parser | second_template | model | str_parser
# 流式输出调用llm
result = chain.stream({"lastname":"张", "gender":"女儿"})

for chunk in result:
    print(chunk, end="", flush=True)

