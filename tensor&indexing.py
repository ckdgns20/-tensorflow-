import tensorflow as tf
import numpy as np


rank_0_tensor = tf.constant(4) #rank_0_tensor 는 0차원의 텐서 형성(스칼라)
print(rank_0_tensor)


rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) #rank_1_tensor 는 1차원의 텐서 형성(벡터)
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16) #rank_2_tensor 는 2차원의 텐서 형성(행렬)
print(rank_2_tensor)


rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],]) #rank_3_tensor 는 3차원의 텐서 형성

print(rank_3_tensor)

rank_4_tensor = tf.zeros([3, 2, 4, 5]) #tf.zeros는 텐서요소를 0으로 채워서 생성

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

#단일 축 인덱싱
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, befor 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy()) # ex) [0:10:1] --> 0이 시작, 10은 끝, 1이 간격 (없을 시([::2]), 기본적으로 [0:1:2] 입력됨)
print("Reverse:", rank_1_tensor[::-1].numpy())

#다축 인덱싱
print(rank_2_tensor[2,0].numpy())
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")
