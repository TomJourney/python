# 提示词加载工具
from poetry_demo.heima_rag.agent_proj.utils.config_handler import prompts_config
from poetry_demo.heima_rag.agent_proj.utils.logger_handler import logger
from poetry_demo.heima_rag.agent_proj.utils.path_tool import get_abs_path


# 加载系统提示词
def load_system_prompts():
    try:
        main_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"yaml配置中没有key=main_prompt_path 的配置项")
        raise e
    try:
        return open(main_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts方法]解析系统提示词报错, {str(e)}")
        raise e


# 加载rag提示词
def load_rag_prompts():
    try:
        rag_summarize_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"yaml配置中没有key=rag_summarize_prompt_path 的配置项")
        raise e
    try:
        return open(rag_summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts方法]解析rag提示词报错, {str(e)}")
        raise e

# 加载report提示词
def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"yaml配置中没有key=report_prompt_path 的配置项")
        raise e
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts方法]解析report提示词报错, {str(e)}")
        raise e

# 测试案例
if __name__ == "__main__":
    # logger.info(load_system_prompts())
    # logger.info(load_rag_prompts())
    logger.info(load_report_prompts())

