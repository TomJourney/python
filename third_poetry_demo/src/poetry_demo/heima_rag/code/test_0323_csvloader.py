from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="../data/stu.csv",
    csv_args={
        "delimiter" : ",", # 指定分隔符
        "quotechar": '"', # 指定带有分隔符文本的引号是单引号还是双引号
        "fieldnames": ['a', 'b', 'c', 'd'], # 或有，指定表头(但原文件的第一行的表头会被当做数据处理)
    },
    encoding="utf-8"  # 指定编码为utf-8
)

# 方式1：批量加载： .load() -> [Document, Document, ...]
documents = loader.load()

for document in documents:
    print("="*20)
    print(type(document), document)

# 方式2： 懒加载 .lazy_load()  迭代器[Document]
print("\n\n", "="*20, "方式2： 懒加载")
for document in loader.lazy_load():
    print("=" * 20)
    print(document)