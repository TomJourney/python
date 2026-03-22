# PyPDFLoader-pdf文件加载器
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="../data/json_line_stu_list.json",
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("=" * 20, i)