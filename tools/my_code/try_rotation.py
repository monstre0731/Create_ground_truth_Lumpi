import numpy as np

# 创建一个示例的 NumPy 数组
arr = np.array([1.2, 2.345678, 3.4, 4.56789])

# 保留小数点后两位，不添加零
formatted_arr = np.array([float(f"{x:.2f}") for x in arr])

print(formatted_arr)
