# agent工具
import random

from langchain_core.tools import tool
from poetry_demo.heima_rag.agent_proj.rag.rag_summarize_service import RagSummarizeService

rag = RagSummarizeService
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str :
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI指数21，最近6小时降雨概率降低"

@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str :
    return random.choice(["深圳", "成都", "重庆"])

@tool(description="获取用户的id，以纯字符串形式返回")
def get_user_id() -> str :
    return random.choice(user_ids)

