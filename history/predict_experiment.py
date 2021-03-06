# -*- coding: utf-8 -*-

from __future__ import print_function

import keras
import time
import warnings
import numpy as np
import pandas as pd
import time
from keras import regularizers
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import keras.backend as K
from sklearn.metrics import roc_auc_score
from keras import optimizers
from sklearn import svm
import csv

warnings.filterwarnings("ignore")


def load_data(filename, seq_len, normalise_window):
    price_data = pd.read_csv(filename, index_col=0, header=0, sep=',', parse_dates=True)
    yy_data = list(price_data['open'])

    row_split = 0.9
    open_data = list(price_data['open'])
    close_data = list(price_data['close'])
    bt_data = list(price_data['bt'])
    vo_data = list(price_data['volume'])

    x_datas = [open_data, close_data, bt_data, vo_data]
    sequence_length = seq_len + 1
    v_num = len(x_datas)
    x_trains = []
    x_tests = []
    for i in range(v_num):
        x_data = x_datas[i]
        x_result = []
        for index in range(len(x_data) - sequence_length +1):
            x_result.append(x_data[index: index + seq_len])  #得到长度为seq_len的向量
        print('result len:',len(x_result))
        print('result shape:',np.array(x_result).shape)
        print(len(x_result[-1]))
        if normalise_window:
            x_result = normalise_windows(x_result)
        x_result = np.array(x_result)        
        row = round(row_split * x_result.shape[0])
        
        x1_train = x_result[:row, :]
        x_trains.append(x1_train)
        
        x1_test = x_result[row:,:]
        x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))
        x_tests.append(x1_test)

    y_result = []
    for index in range(len(yy_data) - sequence_length + 1):
        y_result.append(1 if (yy_data[index + seq_len]-yy_data[index+seq_len-1]) > 0 else 0)
    print('result len:', len(y_result))
    print('1:', y_result.count(1))
    print('0:', y_result.count(0))
    print('result shape:', np.array(y_result).shape)
    
    y_result = np.array(y_result)
    row = round(row_split * y_result.shape[0])
    y_train = y_result[:row]
    x_trains.append(y_train)
    
    y_test = y_result[row:]
    y_test = to_categorical(y_test)
    
    result = zip(*x_trains)
    #np.random.shuffle(result)
    zip_res = list(zip(*result))
    all_train = []
    for i in range(len(zip_res)-1):
        train = list(zip_res[i])
        #train.extend(train)
        #train.extend(train)
        #train.extend(train)
        train = np.array(train) 
        x_train = np.reshape(train, (train.shape[0], train.shape[1], 1))
        all_train.append(x_train)
    y_train = list(zip_res[-1])
    #y_train.extend(y_train)
    #y_train.extend(y_train)
    #y_train.extend(y_train)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train)

    
    return all_train, y_train, x_tests, y_test

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[-1])) - 1) for p in window]
        #normalised_window = [((float(p) - np.mean(window)) / np.std(window)) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers, input_nums):
    model_list = []
    print(layers)
    '''
    for i in range(input_nums):
        model = Sequential()
        model.add(LSTM(layers[1],W_regularizer=regularizers.l2(0.01), return_sequences=False, input_shape=(seq_len,1)))
        model.add(Dropout(0.2))
        model_list.append(model)
    if input_nums > 1:
        model = Sequential()
        model.add(Merge(model_list, mode='concat'))
        #model.add(keras.layers.Add()(model_list))
        
    '''
    model = Sequential()
    model.add(LSTM(layers[1],W_regularizer=regularizers.l2(0.01), return_sequences=False, input_shape=(seq_len,input_nums)))
    model.add(Dropout(0.2))
    
    model.add(Dense(layers[2],W_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(layers[3],W_regularizer=regularizers.l2(0.01), activation='softmax'))
    
    start = time.time()
    rmsprop = optimizers.RMSprop()
    model.compile(loss="categorical_crossentropy", optimizer=rmsprop,metrics=['categorical_accuracy'])
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

def test_auc(predict, y_test):
    y_pred = [da[1] for da in predict]
    #y_true = [da[1] for da in y_test]
    y_true = [np.argmax(da) for da in y_test]
    print('auc:',roc_auc_score(y_true, y_pred))

def check_direction(predict,y_test):
    true_pos = false_pos = true_neg = false_neg = 0
    all_len = len(predict)
    for i in range(all_len):
        pre = np.argmax(predict[i])
        real = np.argmax(y_test[i])
        if pre == 1 and real == 1:
            true_pos += 1
        elif pre == 1 and real == 0:
            false_pos += 1
        elif pre == 0 and real == 0:
            true_neg += 1
        else:
            false_neg += 1

    print('true_pos, false_pos, true_neg, false_neg', true_pos, false_pos, true_neg, false_neg)
    if (true_pos + false_pos > 0):
        pos_pre = true_pos * 1.0 / (true_pos + false_pos)  
    else:
        pos_pre = 0
    if (true_pos + false_neg > 0):
        pos_rec = true_pos * 1.0 / (true_pos + false_neg)
    else:
        pos_rec = 0
    if (true_neg + false_neg > 0):
        neg_pre = true_neg * 1.0 / (true_neg + false_neg)
    else:
        neg_pre = 0
    if (true_neg + false_pos > 0):
        neg_rec = true_neg * 1.0 / (true_neg + false_pos)
    else:
        neg_rec = 0
    accuracy = (true_pos + true_neg) * 1.0 / all_len
    if pos_pre + pos_rec > 0:
        pos_f = 2 * pos_pre * pos_rec / (pos_pre + pos_rec)
    else:
        pos_f = 0
    if neg_pre + neg_rec > 0:
        neg_f = 2 * neg_pre * neg_rec / (neg_pre + neg_rec)
    else:
        neg_f = 0
    print('pos_pre, pos_rec, pos_f', pos_pre, pos_rec, pos_f)
    print('neg_pre, neg_rec, neg_f', neg_pre, neg_rec, neg_f)
    print('accuracy', accuracy)
    res = ['lstm', pos_pre, pos_rec, pos_f, neg_pre, neg_rec, neg_f, accuracy]
    return res

    

def make_svm(x_train, y_train, x_test, y_test):
    all_train = list(zip(*x_train))
    fix_x_train = []
    fix_y_train = []
    for i in range(len(all_train)):
        data = all_train[i] #(3*10*1)
        item = []
        for da in data:
            item.extend(da[:,0])
        fix_x_train.append(item)
        fix_y_train.append(np.argmax(y_train[i]))
    
    all_test = list(zip(*x_test))
    fix_x_test = []
    fix_y_test = []
    for i in range(len(all_test)):
        data = all_test[i]
        item = []
        for da in data:
            item.extend(da[:,0])
        fix_x_test.append(item)
        fix_y_test.append(np.argmax(y_test[i]))

    clf = svm.SVC()
    clf.fit(fix_x_train, fix_y_train)
    y_predict = clf.predict(fix_x_test)
    y_pred = to_categorical(y_predict)
    #print(y_predict, y_pred)
    items = check_direction(y_pred, y_test)
    test_auc(y_pred, y_test)
    return items

def write_result(items):
    with open('expe.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'pos_pre', 'pos_rec', 'pos_f', 'neg_pre', 'neg_rec', 'neg_f', 'accuracy'])
        writer.writerows(items)

if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 100
    seq_len = 10
    batch = 32

    print('> Loading data... ')

    x_train, y_train, x_test, y_test = load_data('../sentiment/daywithprice.csv', seq_len, True)
    for train_set in x_train:
        print('train_set shape:',train_set.shape)  #(3709L, 50L, 1L)
    print('y_train shape:',y_train.shape)  #(3709L,)
    for test_set in x_test:
        print('test_set shape:',test_set.shape)    #(412L, 50L, 1L)
    print('y_test shape:',y_test.shape)    #(412L,)
    print('> Data Loaded. Compiling...')
   
    ''' 
    model1 = build_model([1, seq_len, 2*seq_len, 1], 1)
    model1.fit(x1_train,y_train,batch_size=batch,nb_epoch=epochs,validation_split=0.05)
    point_by_point_predictions_1 = predict_point_by_point(model1, x1_test)
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions_1).shape)  #(412L)
    check_direction(point_by_point_predictions_1,y_test)
    print('Training duration (s) : ', time.time() - global_start_time)
    '''
    ''' 
    #model2 = build_two_model([1, 2*seq_len, 2*seq_len, 1])
    #model2.fit([x1_train, x2_train],y_train,batch_size=batch,nb_epoch=epochs,validation_split=0.05)
    #point_by_point_predictions_2 = predict_point_by_point(model3, [x1_test, x2_test])
    #print('point_by_point_predictions shape:',np.array(point_by_point_predictions_2).shape)  #(412L)
    #check_direction(point_by_point_predictions_2,y_test)
    #print('Training duration (s) : ', time.time() - global_start_time)
    '''
    print(len(x_train))
    test_train = np.zeros((x_train[0].shape[0],seq_len,len(x_train)))
    #print(test_train)
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            #print(x_train[i][j,:,0])
            test_train[j,:,i] = x_train[i][j,:,0]
    #print(test_train)
    
    model3 = build_model([1, 1*seq_len, 3*seq_len, 2], len(x_train))
    model3.fit(test_train,y_train,batch_size=batch,nb_epoch=epochs,validation_split=0.05)
    #print('training_set...')
    #point_by_point_predictions_3 = predict_point_by_point(model3, x_train)
    #print('point_by_point_predictions shape:',np.array(point_by_point_predictions_3).shape)  #(412L)
    #check_direction(point_by_point_predictions_3,y_train)
    #test_auc(point_by_point_predictions_3, y_train)
    print('testing set...')
    test_test = np.zeros((x_test[0].shape[0],seq_len,len(x_test)))
    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            #print(x_test[i][j,:,0])
            test_test[j,:,i] = x_test[i][j,:,0]
    #print(test_test)

    point_by_point_predictions_3 = predict_point_by_point(model3, test_test)
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions_3).shape)  #(412L)
    items_1 = check_direction(point_by_point_predictions_3,y_test)
    test_auc(point_by_point_predictions_3, y_test)
    print('Training duration (s) : ', time.time() - global_start_time)
    items_2 = make_svm(x_train, y_train, x_test, y_test)
    #write_result([items_1, items_2])
    #multiple_predictions = predict_sequences_multiple(model, X_test, seq_len, prediction_len=50)
    #print('multiple_predictions shape:',np.array(multiple_predictions).shape)   #(8L,50L)
    
    #full_predictions = predict_sequence_full(model, X_test, seq_len)
    #print('full_predictions shape:',np.array(full_predictions).shape)    #(412L,)
    
