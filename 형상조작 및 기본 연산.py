import tensorflow as tf

#형상 조작
x = tf.constant([[1],[2],[3]])
#print(x.shape)
#print(x.shape.as_list())

reshaped = tf.reshape(x, [1,3])
print(x.shape)
print(reshaped)
print(reshaped.shape)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)

reshaped_2 = tf.reshape(rank_2_tensor, [-1]) #1차원 형태로 변환
print(reshaped_2)

reshaped_3 = tf.reshape(rank_2_tensor, [2, 3])
print(reshaped_3)
reshaped_4 = tf.reshape(rank_2_tensor, [2, -1]) #-1를 입력시키면 형태를 보고 알아서 변환
print(reshaped_4)

x = tf.reshape(x, [3, 1])
y = tf.range(1, 5)
print(x)
print(y)
print(tf.multiply(x, y)) # multyply와 x*y는 요소별 곱

#텐서 플로우 기본 연산
mat1 = tf.constant([[1,2],
                    [3,4]])
mat2 = tf.constant([[5,6],
                    [7,8]])

product =tf.matmul(mat1, mat2) # matmul 은 행렬곱(multyply 와 x*y 와 다른 것.)
print(product)

element_product = mat1*mat2 #요소별 곱
print(element_product)
element_product2 = tf.multiply(mat1, mat2) #요소별 곱
print(element_product2)
