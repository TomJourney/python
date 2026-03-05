import os
from pathlib import Path

# 1.1 用数据加载器读取简单文本
# 1.1.1 使用langchain读取txt文件，生成Document对象
from langchain_community.document_loaders import TextLoader
file_path = os.path.join(Path.home(), "studynote", "workbench", "python", "poetry-demo", "data", "heishenhua01.txt")
loader = TextLoader(file_path)
documents = loader.load()
print(documents)
# [Document(metadata={'source': '/Users/rong/studynote/workbench/python/poetry-demo/data/heishenhua01.txt'},
# page_content='《黑神话：悟空》的故事可分为六个章节，名为“火照黑云”、“风起黄昏”、“夜生白露”、“曲度紫鸳”、“日落红尘”和“未竟”，并且拥有两个结局，玩家的选择和经历将影响最终的结局。\n\n每个章节结尾，附有二维和三维的动画过场，展示和探索《黑神话：悟空》中的叙事和主题元素。\n\n游戏的设定融合了中国的文化和自然地标。例如重庆的大足石刻、山西省的小西天、南禅寺、铁佛寺、广胜寺和鹳雀楼等，都在游戏中出现。游戏也融入了佛教和道教的哲学元素。')]

