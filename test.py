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
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images-idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels-idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_N_SAMPLE = 10000
#nr.seed(12345)  # random seed 설정

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
        #dataList.append(dataArr.astype('int32'))
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
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return tsDataList, tsLabelList

def cnnTest():
	# 모델 파라미터 불러오기 
    newModel = km.load_model('best_param.h5')

    fd = open("test_output.txt",'w')

    res = newModel.predict(tsDataList,batch_size=_N_SAMPLE)
    # 결과 중에 제일 큰 값을 가져와서 테스트 결과로 사용 
    result = np.argmax(res,axis=1)
    # 비교하기 위해서 정수형으로 변경
    correct = np.argmax(tsLabelList,axis=1)
    # 하나씩 결과와 정답을 비교해서 txt에 출력
    for i in range(_N_SAMPLE):
        if(result[i]==correct[i]):
            fd.write(("%dth result -> %d   correct  ->  %d  == correct!\n")% (i+1,result[i],correct[i]))
        else:
            fd.write(("%dth result -> %d   correct  ->  %d  == correct!\n")% (i+1,result[i],correct[i]))
	# 오류율 검출 
    score = newModel.evaluate(tsDataList,tsLabelList,verbose=0)
    print 'error rate =>',(100-100*score[1])
    fd.write(("error rate -> %.2f \n") % (100-100*score[1]))
    fd.close()

if __name__ == '__main__':
    tsDataList, tsLabelList = loadMNIST()

    cnnTest()
    