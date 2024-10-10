import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense # 구글에 tensorflow.keras.layers.Conv2D 치면 설명 나옴
from tensorflow.keras.optimizers import Adam 
#breakpoint를 누르고 f5를 누르면 각각의 value를 볼 수 있음(이어서 f10을 누르면 한문장씩 돌림)
(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data() #Cifar10 데이터셋 로드
x_train=x_train.astype(np.float32)/255.0 # x_train이 uint8 -> float32로 바뀌고 255를 나눔으로서 범위가 [0,1]로 바뀜
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) #one hot encoding
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=Sequential() 
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3))) 
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu')) 
cnn.add(Conv2D(64,(3,3),activation='relu')) 
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(units=512,activation='relu',))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10,activation='softmax')) # 이 층은 출력층(마지막 층)
# unit=10인 것은 클래스 개수
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001), # 손실함수와 옵티마이저 지정
            metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test))
#fit은 모델을 훈련시켜주는 것(저 안에 축약되어있음)

print('정확률=', cnn.evaluate(x_test,y_test,verbose=0)[1]*100)

plt.plot(hist.history['accuracy'],'r--')
plt.plot(hist.history['val_accuracy'],'r')
plt.ylim((0.2,1.0))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid()
plt.show()
