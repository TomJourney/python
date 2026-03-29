from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

# 定义查询天气工具
@tool(description="查询天气")
def get_weather() -> str :
    return "晴天"


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weather],  # 向智能体提供工具列表
    system_prompt="你是一个聊天助手，可以回答用户问题"
)

result = agent.invoke(
    {
        "messages": [
            {"role":"user", "content":"明天深圳的天气如何"},
        ]
    }
)
# print(result)
# 案例1： 工具列表tools为空的回答：
# AIMessage(content='我无法提供实时或未来的天气信息。建议你通过以下方式查询明天深圳的天气：\n\n- 打开手机上的天气应用（如“天气”或“Weather”）
# \n- 在浏览器中搜索“深圳天气”  \n- 使用权威天气网站，如中国气象局官网、中央气象台、或第三方平台如墨迹天气、彩云天气等\n\n通常这些渠道会提供详细的温度、降水概率、风速和空气质量等信息。希望你有个愉快的一天！'

# 案例2： 工具列表tools有get_weather的回答：
# AIMessage(content='明天深圳的天气是晴天，适合外出活动，记得做好防晒哦！'

# 案例3： 打印大模型回复
for msg in result["messages"]:
    print(type(msg).__name__, ": ",msg.content)
# HumanMessage :  明天深圳的天气如何
# AIMessage :
# ToolMessage :  晴天
# AIMessage :  明天深圳的天气是晴天，适合外出活动，记得做好防晒哦！

