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
from keras.utils.np_utils import to_categorical
import keras.backend as K

warnings.filterwarnings("ignore")

def check_direction(predict,y_test):
    correct = sum([np.argmax(pre) == np.argmax(real) for pre, real in zip(predict, y_test)])
    #correct = sum([ pre_seri[i]*real_seri[i] >= 0 for i in range(len(pre_seri))])
    rate = correct * 1.0 / len(y_test)
    print('correct:', rate)
    return rate

def load_data(filename, seq_len, normalise_window):
    price_data = pd.read_csv(filename, index_col=0, header=0, sep=',', parse_dates=True)
    close_data = list(price_data['open'])
    bt_data = list(price_data['bz'])
    #print('data',data)
    #print('data len:',len(data))
    #print('sequence len:',seq_len)

    sequence_length = seq_len + 1
    close_result = []
    y_result = []
    for index in range(len(close_data) - sequence_length + 1):
        close_result.append(close_data[index: index + seq_len])  #得到长度为seq_len的向量，最后一个作为label
        y_result.append(1 if (close_data[index + seq_len]-close_data[index+seq_len-1]) > 0 else 0)

    bt_result = []
    for index in range(len(bt_data) - sequence_length +1):
        bt_result.append(bt_data[index: index + seq_len])  #得到长度为seq_len的向量
    print('result len:',len(close_result),len(bt_result), len(y_result))
    print('result shape:',np.array(close_result).shape, np.array(bt_result).shape, np.array(y_result).shape)
    print(len(close_result[-1]), len(bt_result[-1]))

    if normalise_window:
        close_result = normalise_windows(close_result)
        bt_result = normalise_windows(bt_result)

    #print(result[:1])
    #print('normalise_windows result shape:',np.array(result).shape)

    close_result = np.array(close_result)
    bt_result = np.array(bt_result)
    y_result = np.array(y_result)
    #划分train、test
    row = round(0.9 * close_result.shape[0])
    train = close_result[:row, :]
    bt_train = bt_result[:row, :]
    y_train = y_result[:row]
    result = zip(train, bt_train, y_train)
    np.random.shuffle(result)
    train, bt_train, y_train = zip(*result)
    train = np.array(train) 
    x1_train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    bt_train = np.array(bt_train) 
    x2_train = np.reshape(bt_train, (bt_train.shape[0], bt_train.shape[1], 1))
    y_train = np.array(y_train)
    #y_train = y_train.astype(int)
    y_train = to_categorical(y_train)
    
    x1_test = close_result[row:, :]
    x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))  
    x2_test = bt_result[row:,:]
    x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1], 1))  
    y_test = y_result[row:]
    #y_test = y_test.astype(int)
    y_test = to_categorical(y_test)

    return [x1_train, x2_train, y_train, x1_test, x2_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[-1])) - 1) for p in window]
        #normalised_window = [((float(p) - np.mean(window)) / np.std(window)) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):  #layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    #model.add(Dense(output_dim=layers[3]))
    model.add(Dense(2,activation="softmax"))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=['categorical_accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_two_model(layers):  #layers [1,50,100,1]

    model1 = Sequential()
    model1.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=False))
    model1.add(Dropout(0.2))

    model2 = Sequential()
    model2.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=False))
    model2.add(Dropout(0.2))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    model.add(Dense(layers[2],activation='relu'))
    model.add(Dense(2,activation='softmax'))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=['categorical_accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    #predicted = np.reshape(predicted, (predicted.size,))
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
    epochs  = 15
    seq_len = 30
    batch = 64

    print('> Loading data... ')

    x1_train, x2_train, y_train, x1_test, x2_test, y_test = load_data('sentiment/daywithprice.csv', seq_len, True)
    print('x1_train shape:',x1_train.shape)  #(3709L, 50L, 1L)
    print('x2_train shape:',x2_train.shape)  #(3709L, 50L, 1L)
    print('y_train shape:',y_train.shape)  #(3709L,)
    print('x1_test shape:',x1_test.shape)    #(412L, 50L, 1L)
    print('x1_test shape:',x2_test.shape)    #(412L, 50L, 1L)
    print('y_test shape:',y_test.shape)    #(412L,)
    print('> Data Loaded. Compiling...')
   
    ''' 
    model1 = build_model([1, seq_len, 2*seq_len, 1])
    model1.fit(x1_train,y_train,batch_size=batch,nb_epoch=epochs,validation_split=0.05)
    point_by_point_predictions_1 = predict_point_by_point(model1, x1_test)
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions_1).shape)  #(412L)
    #check_direction(point_by_point_predictions_1,y_test)
    #print('Training duration (s) : ', time.time() - global_start_time)
    '''
    model2 = build_two_model([1, seq_len, 2*seq_len, 1])
    model2.fit([x1_train, x2_train],y_train,batch_size=batch,nb_epoch=epochs,validation_split=0.05)

    point_by_point_predictions_2 = predict_point_by_point(model2, [x1_test, x2_test])
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions_2).shape)  #(412L)
    check_direction(point_by_point_predictions_2,y_test)
    #print('Training duration (s) : ', time.time() - global_start_time)
    
    #multiple_predictions = predict_sequences_multiple(model, X_test, seq_len, prediction_len=50)
    #print('multiple_predictions shape:',np.array(multiple_predictions).shape)   #(8L,50L)
    
    #full_predictions = predict_sequence_full(model, X_test, seq_len)
    #print('full_predictions shape:',np.array(full_predictions).shape)    #(412L,)

