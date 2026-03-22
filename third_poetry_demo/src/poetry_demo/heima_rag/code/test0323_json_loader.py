from langchain_community.document_loaders import JSONLoader

print("========== 【案例1】 使用jq schema抽取json文件 ")
loader = JSONLoader(
    file_path="../data/stu.json",
    # jq_schema=".name",
    # jq_schema=".other.addr"
    jq_schema=".",  # 抽取整个json文件
    text_content=False, # 告知JSONLoader，抽取的内容不是字符串
)
document = loader.load()
print(document)

print("========== 【案例2】使用jq schema抽取json 列表 文件 ")
loader = JSONLoader(
     file_path="../data/stu_list.json",
    jq_schema=".[].name", # 仅抽取数组的name属性
    text_content=False,  # 告知JSONLoader，抽取的内容不是字符串
)
document = loader.load()
print(document)

print("==========【案例3】 使用jq schema抽取json_lines 文件 ")
loader = JSONLoader(
    file_path="../data/json_line_stu_list.json",
    jq_schema=".name", # 仅抽取数组的name属性
    text_content=False,  # 告知JSONLoader，抽取的内容不是字符串
    json_lines=True   # 告知JSONLoader， 这是一个jsonlines文件（每一行都是一个标准的json对象）
)
document = loader.load()
print(document)