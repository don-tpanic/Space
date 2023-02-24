import numpy as np 

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# print(array.shape)
# print(array)
# print(array.reshape((2, 6)))

array3 = array.reshape((2, 2, 3))
print(array3[0])    # 0th loc, both rots
print(array3[0][0]) # 0th loc, 0th rot
print(array3[0][1]) # 0th loc, 1st rot