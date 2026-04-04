from langchain.agents import create_agent

from poetry_demo.heima_rag.agent_proj.model.model_factory import chat_model
from poetry_demo.heima_rag.agent_proj.utils.prompts_loader import load_system_prompts
from poetry_demo.heima_rag.agent_proj.tools.agent_tools import ( rag_summarize, get_weather, get_user_location,
                                                                 get_user_id, get_current_month, fetch_external_data, fill_context_for_report)
from poetry_demo.heima_rag.agent_proj.tools.agent_middleware import monitor_tool, log_before_model, report_prompt_switch

# 基于react模型的agent
class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages":[
                {"role":"user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report":False}):
            lastest_message = chunk["messages"][-1]
            # 如果 lastest_message.content 有值，才返回
            if lastest_message.content:
                yield lastest_message.content.strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(chunk, end="", flush=True)

    # for chunk in agent.execute_stream("给我生成我的使用报告"):
    #     print(chunk, end="", flush=True)