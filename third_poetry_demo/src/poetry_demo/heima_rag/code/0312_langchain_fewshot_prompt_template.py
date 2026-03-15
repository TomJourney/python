# langchain少样本提示词模版
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi

example_prompt_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

example_data = [
    {"word":"大", "antonym":"小"},
    {"word":"上", "antonym":"下"}
]

# 少样本提示词模板FewShotPromptTemplate， 封装了通用模板对象PromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt_template, # 提示词模板
    examples=example_data,  # 示例数据 ，用于注入动态数据 ， list内套字典
    prefix="返回给定词的反义词，有如下示例：", # 示例数据之前的提示词
    suffix="基于示例告诉我，{input_word}的反义词是什么",  # 示例数据之后的提示词
    input_variables=['input_word']
)

# 获得最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word":"左"})
print(prompt_text.to_string())
# 单词：大, 反义词：小
# 单词：上, 反义词：下
# 基于示例告诉我，左的反义词是什么

# 调用大模型
model = Tongyi(model="qwen-max")
result = model.invoke(input=prompt_text)
print(result)  # 基于您给出的示例，"左" 的反义词是 "右"。
