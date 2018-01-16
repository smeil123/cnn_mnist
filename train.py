# -*- coding: utf-8 -*-
import os
import os.path as op
import struct as st
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import keras.utils as np_utils
import pydot
import time

# MNIST 데이터 경로
_SRC_PATH = u'mnist\\raw_binary'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images-idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels-idx1-ubyte'
# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_N_SAMPLE = 60000
nr.seed(12345)  # random seed 설정
maxEpoch = 10

def loadData(fn):
    print 'loadData', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    print 'nRow', nRow
    print 'nCol', nCol
    
    # data: unsigned byte
    dataList = np.ones((_N_SAMPLE,_N_ROW,_N_COL,1))

    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(1,28,28,1)
        dataList[i]=dataArr
    fd.close()
    
    print 'done.'
    print
    
    return dataList/255.0

def loadLabel(fn):
    print 'loadLabel', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print 'done.'
    print
    labelList = np_utils.to_categorical(labelList,10)
    return labelList

def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    return trDataList, trLabelList


def cnnTest():
	fd = open("train_log.txt","w")
	# 모델 구성(2(input) -> CONV(ReLU) -> CONV(ReLU) -> FC(sigmoid))
	model = km.Sequential()
	# 입력 (None,28,28,1)
	# 출력 (None,28,28,32)
	model.add(kl.Conv2D(input_shape=(28, 28, 1), filters=32,
	                    kernel_size=(3, 3), strides=1,
	                    padding='same'))    # zero-padding
	model.add(kl.Activation('relu'))
	# 입력 (None,28.28,32)
	# 출력 (None,26,26,32)
	model.add(kl.Conv2D(filters=32,
	                    kernel_size=(3, 3), strides=1))
	model.add(kl.Activation('relu'))
	# 입력 (None,26,26,32)
	# 출력 (None,13,13,32) 
	model.add(kl.MaxPooling2D(pool_size=(2,2)))
	model.add(kl.Dropout(0.25))
	# 출력 (None,13*13*32)
	model.add(kl.Flatten())
	model.add(kl.Dense(128,activation='relu'))
	model.add(kl.Dropout(0.5))
	model.add(kl.Dense(10,activation='softmax'))

	# 학습 설정
	model.compile(loss='categorical_crossentropy',
	               optimizer='adam', metrics=['accuracy'])

	# 모델 구조 그리기
	ku.plot_model(model, 'model.png')

	# 학습(10회 반복, 10000개 샘플씩 배치 학습)
	for epoch in range(maxEpoch):
		#trLoss = model.train_on_batch(trDataList,trLabelList)
		model.fit(trDataList,trLabelList,batch_size=10000,epochs=1)
		score = model.evaluate(trDataList,trLabelList,verbose=0)
		fd.write(("Epoch %d/%d, error rate : %f\n")%(epoch+1,maxEpoch,100-score[1]*100))

	fd.close()

	km.save_model(model,'best_param.h5')

if __name__ == '__main__':
    start = time.time()
    trDataList, trLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)

    cnnTest()
    
    end = time.time() - start

    print 'time : ', end
