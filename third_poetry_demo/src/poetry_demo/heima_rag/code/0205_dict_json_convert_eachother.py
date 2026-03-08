# python字典与json相互转换
import json

dict = {
    "name":"张三",
    "age":"11",
    "gender":"男"
}

print("\n========== python字典或字典列表转为json对象或json数组 \n")
# 转为json字符串
json_str = json.dumps(dict, ensure_ascii=False)
print(json_str) # {"name": "张三", "age": "11", "gender": "男"}

# 【2】 python字典列表转为json字符串
arr_dict = [
    {
    "name":"张三",
    "age":"3",
    "gender":"男"
    },
    {
    "name":"李四",
    "age":"4",
    "gender":"女"
    }
]
# 转为json数组
json_arr = json.dumps(arr_dict, ensure_ascii=False)
print(json_arr)
# [{"name": "张三", "age": "3", "gender": "男"}, {"name": "李四", "age": "4", "gender": "女"}]

# ========== json对象或数组转为python字典或列表
print("\n\n==========json对象或数组转为python字典或列表\n")
temp_json_str = '{"name": "张三", "age": "11", "gender": "男"}'
temp_json_arr = '[{"name": "张三", "age": "3", "gender": "男"}, {"name": "李四", "age": "4", "gender": "女"}]'

# 转为字典或数组
temp_dict = json.loads(temp_json_str)
temp_dict_arr = json.loads(temp_json_arr)

print(temp_dict, type(temp_dict))
# {'name': '张三', 'age': '11', 'gender': '男'} <class 'dict'>
print(temp_dict_arr, type(temp_dict_arr))
# [{'name': '张三', 'age': '3', 'gender': '男'}, {'name': '李四', 'age': '4', 'gender': '女'}] <class 'list'>




