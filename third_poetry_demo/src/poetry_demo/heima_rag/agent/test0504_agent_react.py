from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

# 获取股票价格工具或函数
@tool(description="获取体重，返回值是整数，单位千克")
def get_weight(name: str) -> str:
    return 90

# 获取股票信息
@tool(description="获取身高，返回值是整数，单位厘米")
def get_height(name: str) -> str:
    return 172

agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weight, get_height],  # 向智能体提供工具列表
    system_prompt="""你是严格遵循ReAct框架的智能体，必须按照[思考->行动->观察->再思考]的流程解决问题。
    且**每轮仅能思考并调用1个工具**，禁止单次调用多个工具。
    并告诉我你的思考过程，工具的调用原因。按思考、行动、观察三个结构告知我
    """
)

for chunk in agent.stream(
    {"messages":[{"role":"user", "content":"计算我的BMI"}]},
    stream_mode="values"
):
    latest_msg = chunk['messages'][-1]

    if latest_msg.content:
        print(type(latest_msg).__name__, latest_msg.content)

    try:
        if latest_msg.tool_calls:
            print(f"工具调用：{ [tc['name'] for tc in latest_msg.tool_calls] }")
    except AttributeError as e:
        pass

# HumanMessage 计算我的BMI
# AIMessage [思考]
# 要计算BMI（身体质量指数），需要知道用户的体重（千克）和身高（厘米）。公式为：
# $$ \text{BMI} = \frac{\text{体重（kg）}}{(\text{身高（m）})^2} $$
# 因此，首先需要获取用户的体重或身高。由于两个数据都未知，我先选择获取体重。
#
# [行动]
#
# 工具调用：['get_weight']
# ToolMessage 90
# AIMessage [观察]
# 获取到用户的体重为90千克。
#
# [思考]
# 现在已知体重为90千克，接下来需要获取用户的身高（厘米），以便计算BMI。因此，下一步调用获取身高的工具。
#
# [行动]
#
#
# 工具调用：['get_height']
# ToolMessage 172
# AIMessage [观察]
# 获取到用户的身高为172厘米。
#
# [思考]
# 现在已知体重为90千克，身高为172厘米。根据BMI公式：
# $$ \text{BMI} = \frac{90}{(1.72)^2} \approx 30.3 $$
# 计算得出用户的BMI约为30.3，属于肥胖范围（BMI ≥ 30）。
