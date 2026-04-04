"""
yaml
k:v
"""

import yaml

from poetry_demo.heima_rag.agent_proj.utils.path_tool import get_abs_path

# 加载 rag 配置
def load_rag_config(config_path:str=get_abs_path("config/rag.yaml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 加载 rag 配置
def load_chroma_config(config_path:str=get_abs_path("config/chroma.yaml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 加载 rag 配置
def load_prompts_config(config_path:str=get_abs_path("config/prompts.yaml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 加载 rag 配置
def load_agent_config(config_path:str=get_abs_path("config/agent.yaml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 变量
rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

# 测试案例
if __name__ == "__main__":
    print(rag_config["chat_model_name"]) # qwen3-max