# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import warnings
import numpy as np
import pandas as pd
import time
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.models import Sequential

warnings.filterwarnings("ignore")

def check_direction(predict,y_test):
    pre_seri = pd.Series(predict)
    real_seri = pd.Series(y_test)
    pre_var = pre_seri - pre_seri.shift(1)
    real_var = real_seri - real_seri.shift(1)
    correct = sum([ pre_var[i]*real_var[i] > 0 for i in range(1,len(pre_var))])
    rate = correct * 1.0 / (len(real_var) - 1)
    print('correct:', rate)
    return rate

def load_data(filename, seq_len, normalise_window):
    price_data = pd.read_csv(filename, index_col=0, header=0, sep=',', parse_dates=True)
    data = list(price_data['close'])
    #print('data',data)
    #print('data len:',len(data))
    #print('sequence len:',seq_len)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length +1):
        #print(index, index+sequence_length)
        result.append(data[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label

    print('result len:',len(result))
    print('result shape:',np.array(result).shape)
    print(len(result[-1]))

    if normalise_window:
        result = normalise_windows(result)

    #print(result[:1])
    #print('normalise_windows result shape:',np.array(result).shape)

    result = np.array(result)

    #划分train、test
    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):  #layers [1,50,100,1]

    model1 = Sequential()
    model1.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model1.add(Dropout(0.2))

    model1.add(LSTM(layers[2],return_sequences=False))
    model1.add(Dropout(0.2))
    
    model2 = Sequential()
    model2.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model2.add(Dropout(0.2))

    model2.add(LSTM(layers[2],return_sequences=False))
    model2.add(Dropout(0.2))
    

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

#滚动预测
def predict_sequence_full(model, data, window_size):  #data X_test
    curr_frame = data[0]  #(50L,1L)
    predicted = []
    for i in xrange(len(data)):
        #x = np.array([[[1],[2],[3]], [[4],[5],[6]]])  x.shape (2, 3, 1) x[0,0] = array([1])  x[:,np.newaxis,:,:].shape  (2, 1, 3, 1)
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])  #np.array(curr_frame[newaxis,:,:]).shape (1L,50L,1L)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)   #numpy.insert(arr, obj, values, axis=None)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):  #window_size = seq_len
    prediction_seqs = []
    for i in xrange(len(data)/prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in xrange(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 2
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = load_data('index/day.csv', seq_len, True)
    print('X_train shape:',X_train.shape)  #(3709L, 50L, 1L)
    print('y_train shape:',y_train.shape)  #(3709L,)
    print('X_test shape:',X_test.shape)    #(412L, 50L, 1L)
    print('y_test shape:',y_test.shape)    #(412L,)

    print('> Data Loaded. Compiling...')
    model = build_model([1, 50, 100, 1])
    model.fit([X_train, X_train],y_train,batch_size=512,nb_epoch=epochs,validation_split=0.05)

    #multiple_predictions = predict_sequences_multiple(model, X_test, seq_len, prediction_len=50)
    #print('multiple_predictions shape:',np.array(multiple_predictions).shape)   #(8L,50L)
    
    #full_predictions = predict_sequence_full(model, X_test, seq_len)
    #print('full_predictions shape:',np.array(full_predictions).shape)    #(412L,)

    point_by_point_predictions = predict_point_by_point(model, [X_test, X_test])
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions).shape)  #(412L)
    check_direction(point_by_point_predictions,y_test)
    print('Training duration (s) : ', time.time() - global_start_time)
