"""
为整个工程提供统一的绝对路径 
"""

import os

def get_project_root() -> str:
    """
    获取工程所在的根目录
    """
    # 当前文件的绝对路径 （__file__是python常量，表明当前文件）
    current_file_abs_path = os.path.abspath(__file__)
    # 获取工程的根目录，先获取文件所在文件夹绝对路径
    current_dir = os.path.dirname(current_file_abs_path)
    # 获取工程根目录
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径：返回绝对路径
    :param relative_path:
    :return:
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

if __name__ == "__main__":
    print(get_abs_path("data/故障排除.txt"))
    # /third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/故障排除.txt