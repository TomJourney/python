# 计算余弦相似度
import numpy

# 计算点积
def get_dot(vec_a, vec_b):
    # return vec_a.dot(vec_b)  可以直接计算点积
    if len(vec_a) != len(vec_b):
        raise ValueError("2个向量的维度必须相同")

    dot_sum = 0
    for a, b in zip(vec_a, vec_b):
        dot_sum += a * b
    return dot_sum

# 计算模长
def get_norm(vector):
    sum_square = 0
    for element in vector:
        sum_square += element**2
    # 使用numpy sqrt开根号
    return numpy.sqrt(sum_square)

# 计算余弦相似度
def cos_similarity(vec_a, vec_b):
    result = get_dot(vec_a, vec_b) / ( get_norm(vec_a) * get_norm(vec_b) )
    print(result)

# 测试
if __name__ == '__main__':
    v1 = [0.5, 0.5]
    v2 = [0.7, 0.7]
    cos_similarity(v1, v2) # 1.0