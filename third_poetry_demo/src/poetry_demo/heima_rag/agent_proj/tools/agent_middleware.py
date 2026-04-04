from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from poetry_demo.heima_rag.agent_proj.utils.logger_handler import logger
from poetry_demo.heima_rag.agent_proj.utils.prompts_loader import load_report_prompts, load_system_prompts


# 工具执行的监控
@wrap_tool_call
def monitor_tool(
        # 请求的数据封装（入参）
        request: ToolCallRequest,
        # 执行的函数本身（函数）
        handler: Callable[[ToolCallRequest], ToolMessage | Command ],
) -> ToolMessage | Command:
    logger.info(f"monitor_tool方法：执行工具={request.tool_call['name']}")
    logger.info(f"monitor_tool方法：传入参数={request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"monitor_tool方法：执行工具={request.tool_call['name']}调用成功")

        # 若工具fill_context_for_report被调用，则设置request.runtime.context['report']=True，用于标记该方法被调用
        if request.tool_call['name'] == 'fill_context_for_report':
            request.runtime.context['report'] = True
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")

# 在模型执行前输出日志
@before_model
def log_before_model(
        # 整个agent的状态记录
        state: AgentState,
        # 记录了整个执行过程中的上下文信息
        runtime: Runtime
):
    logger.info(f"log_before_model方法：即将调用模型，带有{len(state['messages'])}条消息")
    logger.info(f"log_before_model方法: {type(state['messages'][-1]).__name__}： {state['messages'][-1].content.strip()}")
    return None

# 动态切换提示词
@dynamic_prompt # 每一次生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get('report', False)

    # 若true，表示是报告生成长，返回报告生成提示词内容
    if is_report:
        print("*" * 20, " 动态切换提示词 ", "*" * 20)
        return load_report_prompts()
    else:
        return load_system_prompts()
