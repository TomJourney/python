# 文本分块
# 分块策略： 按固定字符数分块，递归分块， 基于格式分块，基于版本分块，语义分块， 命题分块；
import os
from pathlib import Path

# p83 2.2.1 langchain的CharacterTExtSplitter 工具
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import  CharacterTextSplitter

file_path = os.path.join(Path.home(), "studynote", "workbench", "python", "poetry-demo", "data", "heishenhua01.txt")
loader = TextLoader()
documents = loader.load()

# 设置分块器： 指定块的大小为50个字符， 无重叠
texxt_splitter = CharacterTextSplitter(
    chunk_size=0,
    chunk_overlap=0
)
# 执行分块
chunks = text
print(documents)

