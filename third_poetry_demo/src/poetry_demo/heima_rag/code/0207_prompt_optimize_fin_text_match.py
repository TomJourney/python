# 提示词优化案例_金融文本匹配

# 1 获取client对象
from openai import OpenAI
import json
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

example_data = {
    "是":[
        ("公司ABC发布了季度财报，显示盈利增长。", "财报披露，公司ABC利润上升")
        , ("公司ITCAST发布了年度财报，显示盈利大幅度增长。", "财报披露，公司ITCAST更赚钱了。")
    ],
    "不是":[
        ("黄金价格下跌，投资者抛售。", "外汇市场交易额创下新高。")
        , ("央行降息，刺激经济增长。", "新能源技术的创新。")
    ]
}

# 问题清单
questions = [
    ("利率上升，影响房地产市场。", "高利率对房地产有一定的冲击"),
    ("油价大幅度下降，能源公司面临挑战", "未来只能城市的建设趋势越加明显。"),
    ("股票市场今日大涨，投资者乐观。", "持续上涨的市场让投资者感到满意。")

]

"""
    {"role":"system", "content":f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："}
    
    {"role":"user", "content":"句子1：[公司ABC发布了季度财报，显示盈利增长。] 句子2:[财报披露，公司ABC利润上升。]"}
    {"role":"assistant", "content":"是"} 
    
    {"role":"user", "content":"句子1：[公司ITCAST发布了年度财报，显示盈利大幅度增长。] 句子2:[财报披露，公司ITCAST更赚钱了。]"}
    {"role":"assistant", "content":"是"} 
    
    {"role":"user", "content":"句子1：[黄金价格下跌，投资者抛售。] 句子2:[外汇市场交易额创下新高。]"}
    {"role":"assistant", "content":"不是"} 
    
    {"role":"user", "content":"句子1：[央行降息，刺激经济增长。] 句子2:[新能源技术的创新。]"}
    {"role":"assistant", "content":"不是"} 
    
    {"role":"user", "content": f"按照上述示例，回答这2个句子的情况。句子1:[...], 句子2:[...]"}
"""

# 整理提示词
print("\n==========整理提示词\n")
messages = [
    {"role":"system", "content":f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："}
]

for key, value in example_data.items():
    for t in value:
        messages.append(
            {"role":"user", "content":f"句子1：[{t[0]}], 句子2：[{t[1]}]"}
        )
        messages.append(
            {"role": "assistant", "content": key}
        )
for x in messages:
    print(x)

print("\n==========整理问题，并向大模型提问\n")
# 整理问题，并向大模型提问
for question in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages = messages + [{"role":"user", "content":f"句子1：[{question[0]}], 句子2：[{question[1]}]"}]
    )
    print(response.choices[0].message.content)

# 是
# 不是
# 是

