from langchain_community.document_loaders import  TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 【案例1】使用TextLoader读取文本文件
print("====== 【案例1】 使用TextLoader读取文本文件：")
loader = TextLoader(
    file_path="../data/python_base_syntax.txt",
)

documents = loader.load()
# print(documents)
# print(len(documents)) # 1

# 【案例2】使用 RecursiveCharacterTextSplitter 分割字符
print("========== 【案例2】使用 RecursiveCharacterTextSplitter 分割字符" )
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 分段的最大字符数
    chunk_overlap=50,  # 分段之间允许的重叠字符数
    separators=["\n\n", "\n", "!", " ", "!"], # 文本自然段落分割的依据符号
    length_function=len # 统计字符的依据函数
)

# 分割文本
split_docs = splitter.split_documents(documents)
print(len(split_docs))
loop_time = 0
for doc in split_docs:
    loop_time = loop_time + 1
    print("="*20, "第" + str(loop_time), "个段落")
    print(doc)
    print("=" * 20)

