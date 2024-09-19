import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(2, 10, num=50) # 2에서 10까지 숫자 50개 생성
Y = np.random.rand(50)*10 + 2 
Y.sort()
print('X = ', X)
print('Y = ', Y)


plt.plot(X, Y, 'ro')
plt.show()

W = tf.Variable(np.zeros(()), name='weight') # 가중치
b = tf.Variable(np.zeros(()), name='bias')   # 편향

def linear_regression(x):   # 입력값 x 에 대해 선형 방정식 정의
    return W*x + b
 
def mean_square(y_pred, y):  # 예측 값(y_pred)과 실제 값(y) 간의 평균 제곱 오차 계산(손실 함수 부분)
    return tf.reduce_mean(tf.square(y_pred - y))

# 학습 파라미터 설정
epochs = 1000   # 훈련 횟수 1000
optimizer = tf.optimizers.SGD() #가중치와 편향을 옵티마이저로 업데이트
 
for epoch in range(1, epochs + 1):
    with tf.GradientTape() as t:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
    
    # compute gradients
    gradients = t.gradient(loss, [W, b])
 
    # update W and b following gradients
    optimizer.apply_gradients(zip(gradients, [W, b]))
 
    if epoch % 50 == 0: # 매 50 에포크마다 현재 손실 값과 가중치,편향을 출력
        print(f'{epoch} epoch : loss = {loss}, W = {W.numpy()}, b = {b.numpy()}')

# 데이터 시각화부분
plt.plot(X, Y, 'ro', label='Origin data')
plt.plot(X, np.array(W*X + b), label='Fitted line')
plt.legend() # 그래프에 범례를 추가
plt.show()