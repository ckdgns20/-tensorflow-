import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

"""
#스칼라에 대한 그래디언트 계산
x = tf.Variable(3.0)   # x를 3.0 으로 초기화

with tf.GradientTape() as tape:  
  y = x**2                       # **은 제곱연산

# dy = 2x * dx
dy_dx = tape.gradient(y, x) #y에 대한 x의 그래디언트(미분)을 계산
dy_dx.numpy()
print(dy_dx)
"""

w = tf.Variable(tf.random.normal((3, 2)), name='w')  #nomal은 표준정규분포에서 난수(랜덤숫자)를 생성해라.
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')

print(w)
print(b)
x = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
  y = x @ w + b # @는 행렬곱을 의미
  loss = tf.reduce_mean(y**2) #위에서 나온 2개의 요소를 제곱해서 평균을 구한 것.
"""
[dl_dw, dl_db] = tape.gradient(loss, [w, b]) #loss를 w 와 b에 대해 미분을 계산해서 dl_dw 와 dl_db에 저장(명시적으로 지정)

print(w.shape)
print(dl_dw.shape)
print(dl_dw)
print(dl_db)
"""
# 아래는 사전을 전달
my_vars = {
    'w': w,
    'b': b
}

grad = tape.gradient(loss, my_vars)
print(grad['b'])
