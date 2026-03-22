# PyPDFLoader-pdf文件加载器
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="../data/python_syntax.pdf",
    # mode="page" # 默认是page模式，每个页面形成一个document文档对象
    mode = "single",  # 不管多少页，只返回一个document对象
    # password="123455"  # pdf文件打开密码
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("=" * 20, f"第{str(i)}个段落")