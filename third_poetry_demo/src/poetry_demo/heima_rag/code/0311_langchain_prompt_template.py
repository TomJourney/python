# langchain通用提示词模版
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

# 提示词模版类PromptTemplate，是Runnable接口的实现类，它可以加入到langchain中的链条
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}， 刚生了{gender}, 你帮我起个名字，简单回答。"
)

# 调用 .format方法注入信息即可
prompt_text = prompt_template.format(lastname="张", gender="女儿")
print("prompt_text = " + prompt_text)
# 我的邻居姓张， 刚生了女儿, 你帮我起个名字，简单回答。

model = Tongyi(model="qwen-max")
result = model.invoke(input=prompt_text)
print(result) # 张家欣