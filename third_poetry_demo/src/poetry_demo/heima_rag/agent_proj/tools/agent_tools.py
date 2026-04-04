# agent工具
import os
import random

from langchain_core.tools import tool
from poetry_demo.heima_rag.agent_proj.rag.rag_summarize_service import RagSummarizeService
from poetry_demo.heima_rag.agent_proj.utils.config_handler import agent_config
from poetry_demo.heima_rag.agent_proj.utils.logger_handler import logger
from poetry_demo.heima_rag.agent_proj.utils.path_tool import get_abs_path

rag = RagSummarizeService()
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", 
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"
]

# 外部数据字典
external_data = {}


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

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)

def generate_external_data():
    """
    {
        "user_id":{
            "month":{"特征":xxx, "效率": xxx, ...}
            "month":{"特征":xxx, "效率": xxx, ...}
            "month":{"特征":xxx, "效率": xxx, ...}
            ...
        },
        "user_id":{
            "month":{"特征":xxx, "效率": xxx, ...}
            "month":{"特征":xxx, "效率": xxx, ...}
            "month":{"特征":xxx, "效率": xxx, ...}
            ...
        }
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                word_list : list[str] = line.strip().split(",")
                # 解析每行字段
                user_id: str = word_list[0].replace('"', "")
                feature: str = word_list[1].replace('"', "")
                efficiency: str = word_list[2].replace('"', "")
                consumption: str = word_list[3].replace('"', "")
                comparison: str = word_list[4].replace('"', "")
                time: str = word_list[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征":feature,
                    "效率":efficiency,
                    "耗材":consumption,
                    "对比":comparison,
                }

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，若未检索到则返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str :
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"fetch_external_data方法：没有检索到用户{user_id}在{month}的使用记录数据")
        return ""

# if __name__ == '__main__':
#     print(fetch_external_data("1001", "2025-01"))
# {'特征': '65㎡公寓 | 单身 | 木地板', '效率': '覆盖率:85%\\n日均清扫:45㎡\\n漏扫区域:沙发底部（高度不足）', '耗材': '主刷寿命:剩余60天\\nHEPA滤网:剩余40%', '对比': '优于65%同面积用户（清洁频率更高）'}


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report方法：本方法被调用"

