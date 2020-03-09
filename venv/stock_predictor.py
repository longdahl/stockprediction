import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Activation,Dropout,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adamax,Adadelta,Adagrad
from keras.layers import LeakyReLU,Dense,Activation,Dropout,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adamax,Adadelta,Adagrad
import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1, 'CPU': 4})
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import tensorflow.python.keras.backend as K
K.set_session(sess)

def standardize(train_features,val_features,test_features):
    scaler = StandardScaler()
    print(type(train_features))
    train_features = scaler.fit_transform(train_features)
    val_features  = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    return train_features, val_features, test_features

def create_train_val_test(data,test_size):

    train,val = np.array(train_test_split(np.array(data),test_size= test_size,shuffle= False))
    train,test =  np.array(train_test_split(np.array(train),test_size= test_size,shuffle= False))

    return train,val,test

def create_train_val(data,val_size):

    train,val = np.array(train_test_split(np.array(data),test_size= test_size,shuffle= False))

    return train,val

def data_preparer(data,test_size):

    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    #remove index columns if present

    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'],axis = 1)

    if 'index' in data.columns:
        data = data.drop(['index'],axis = 1)

    train,val,test = create_train_val_test(data,test_size)

    train_labels = train[:,0]
    val_labels = val[:,0]
    test_labels = test[:,0]

    train_features = np.delete(train, 0, axis=1)
    val_features = np.delete(val, 0, axis=1)

    test_features = np.delete(test, 0, axis=1)

    train_features,val_features,test_features = standardize(train_features,val_features,test_features)

    return train_labels,val_labels,test_labels,train_features,val_features,test_features

def data_preparer_no_test(data,val_size,test_data):
    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    #remove index columns if present

    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'],axis = 1)
    if 'index' in data.columns:
        data = data.drop(['index'],axis = 1)
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(['Unnamed: 0'],axis = 1)

    if 'index' in test_data.columns:
        test_data = test_data.drop(['index'],axis = 1)

    train,val = create_train_val(data,test_size)
    test = np.array(test_data)

    train_labels = train[:,0]
    val_labels = val[:,0]
    test_labels = test[:, 0]

    train_features = np.delete(train, 0, axis=1)
    val_features = np.delete(val, 0, axis=1)
    test_features = np.delete(test, 0, axis=1)

    train_features, val_features, test_features = standardize(train_features, val_features, test_features)

    return train_labels, val_labels, test_labels, train_features, val_features, test_features


def make_model(train_features):
    tf.compat.v1.reset_default_graph()
    K.clear_session()

    with tf.device("/device:GPU:0"):
        model = Sequential()
        model.add(Dropout(0.5,input_shape=(train_features.shape[-1],)))
        model.add(Dense(train_features.shape[1]))
        model.add(BatchNormalization(axis = 1))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.15))
        model.add(Dense(int(train_features.shape[1] / 2)))
        model.add(BatchNormalization(axis = 1))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.15))
        model.add(Dense(1,activation='sigmoid'))

        metrics = ['acc']
        model.compile(
            loss = 'binary_crossentropy',
            optimizer = Adagrad(lr = 0.01),
            #optimizer=Adadelta(lr = 1.0, rho=0.95) #alternative
            metrics = metrics
            )
        return model

def train_single_model(train_features,train_labels,val_features,val_labels,test_features,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE):
    model = make_model(train_features)

    fileP = filepath + modelName + ".h5"

    callbacks = [EarlyStopping(monitor = 'val_loss',patience = PATIENCE),ModelCheckpoint(filepath = fileP,monitor='val_loss',save_best_only=True)]

    model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs = EPOCHS,
        callbacks = callbacks,
        #class_weight = class_weights,
        validation_data = (val_features,val_labels))
    test_predict = model.predict(test_features)

    return test_predict

def make_roc_plot(predicts,test_labels,modelName,filepath):
    fpr,tpr,threshhold = metrics.roc_curve(test_labels,predicts)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title("ROC for " + modelName)
    plt.plot(fpr,tpr,'b',label = 'AUC %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.show
    plt.savefig(filepath + modelName + ".png")
    plt.close()

def save_model_results(predicts,test_labels,modelName,filepath,intervals):
    make_roc_plot(predicts,test_labels,modelName,filepath)
    path = filepath + modelName + ".txt"
    file1 = open(path,"w+")

    prob = [x / intervals for x in range(1,intervals,1)]

    file1.write("Model performance for model " + modelName)
    file1.write("\n")


    for p in prob:
        pred_labels = np.array([0 if x < p else 1 for x in predicts])

        precision,recall,fscore,support = score(test_labels,pred_labels)

        print('fscore at threshhold ' ,p,' for zeros / ones : {}'.format(fscore[1]))
        print('precision at threshhold ', p, ' for zeros / ones : {}'.format(precision[1]))
        print('recall at threshhold ', p, ' for zeros / ones : {}'.format(recall[1]))
        print("\n")

        file1.write('fscore at threshhold ' + str(p) + ' for ones : {}'.format(round(fscore[1],3)))
        file1.write('precision at threshhold ' + str(p) + ' for ones : {}'.format(round(precision[1],3)))
        file1.write('recall at threshhold ' + str(p) + ' for ones : {}'.format(round(recall[1],3)))
        file1.write("\n")
    file1.write("The AUC-score is " + str(auc_score(test_labels,predicts)))
    file1.close()


def simple_bagging(n,train_features,train_labels,val_features,val_labels,test_features,test_labels,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals):
    predicts = []
    for i in range(1,n+1):
        print("training number: ",i)
        test_predict = train_single_model(train_features,train_labels,val_features,val_labels,test_features,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE)
        predicts.append(test_predict)
    predicts = np.sum(predicts,0)
    predicts = predicts / n

    test_labels = np.int32(test_labels)
    predicts = np.float64(predicts)

    save_model_results(predicts,test_labels,modelName,filepath,print_intervals)

    np.savetxt(filepath + modelName + ".csv",predicts,delimiter= ",")

    return predicts

def model_constructor(data,test_data,baggingN,filepath,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals,dataName,test_size):

    #train_labels, val_labels, test_labels, train_features, val_features, test_features = data_preparer(data,test_size)
    train_labels, val_labels, test_labels, train_features, val_features, test_features = data_preparer_no_test(data, test_size,test_data)
    modelName = dataName + "_ANN with num bagging: " + str(baggingN)

    predicts = simple_bagging(baggingN,train_features,train_labels,val_features,val_labels,test_features,test_labels,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals)

    results_file = pd.concat([pd.Series(test_labels),pd.DataFrame(predicts),pd.DataFrame(test_features)],axis = 1)

    results_file.columns = ["test_labels","predicts"] + list(test_data.columns[1::])

    sorted_res = results_file.sort_values('predicts', axis=0, ascending=False)

    results_file.to_pickle(filepath + "Resultsfile:" + modelName + ".txt")

    return predicts,results_file

def trading_simulator(simulation_days,baggingN,filepath,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals,dataName,test_size):

    load_data = pd.read_pickle(filepath + dataName + ".txt")
    last_day = max(load_data['Elapsed_days'])
    for i,sim_day in enumerate(range(last_day - simulation_days,last_day)):
        print("simulation day: ", str(i))
        data = load_data.loc[load_data['Elapsed_days'] <= sim_day]
        test_data = load_data.loc[load_data['Elapsed_days'] == sim_day + 1]
        predicts,results_file = model_constructor(data,test_data,baggingN, filepath, BATCH_SIZE, EPOCHS, PATIENCE, print_intervals, dataName, test_size)
    return predicts


simulation_days = 1
baggingN = 1
filepath = "C:/Users/Mikkel/Desktop/machine learning/stock_prediction/"
BATCH_SIZE = 100
EPOCHS = 20
PATIENCE = 10
print_intervals = 10
dataName = "sp_500_data_2019okt2020feb18_ver2"
test_size = 0.2

results_file = trading_simulator(simulation_days,baggingN,filepath,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals,dataName,test_size)


