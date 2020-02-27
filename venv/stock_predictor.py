import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import metrics
from sklearn.decomposition import PCA
from sklearn.utils import class_weight

from keras.callbacks import Earlystopping,ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Activation,Dropout,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adamax,Adadelta,Adagrad
from keras.layers import LeakyReLU,Dense,Activation,Dropout,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adamax,Adadelta,Adagrad


config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

from keras import backend as K
K.set_session(sess)



def print_prob(test_labels,pred_labels,ynew_prob,intervals):

    prob = [x / intervals for x in range(1,intervals,1)]

    for p in prob:
        pred:labels = np.array([0 if x < p else 1 for x in ynew_prob])

        precision,recall,fscore,support = score(test_labels,pred_labels)

        print('fscore at threshhold ' ,p,' for zeros / ones : {}'.format(fscore[1]))
        print('precision at threshhold ', p, ' for zeros / ones : {}'.format(precision[1]))
        print('recall at threshhold ', p, ' for zeros / ones : {}'.format(recall[1]))
        print("\n")

def make_model(train):
    tf.reset_default_graph()
    K.clear_session()

    with tf.device("/device:GPU:0"):
        model = Sequential()
        model.add(Dropout(0.5,input_shape=(train.shape[-1],)))
        model.add(Dense(train.shape[1]))
        model.add(BatchNormalization(axis = 1))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.15))
        model.add(Dense(int(train.shape[1] / 2)))
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
    model = make_model(train)

    fileP = filepath + modelName + ".h5"

    callbacks = [Earlystopping(monitor = 'val_loss',patience = PATIENCE),ModelCheckpoint(filepath = fileP,monitor='val_loss',save_best_only=True)]

    model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epcchs = EPOCHS,
        callbacks = callbacks,
        #class_weight = class_weights,
        validation_data = (val_features,val_labels))
    test_predict = model.predict(test_features)

    return test_predict

def make_roc_plot(predicts,modelName,filepath):
    fpr,tpr,threshhold = metrics.roc_curve(test_labels,predicts)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title("ROC for " + modelName)
    plt.plot(fpr,tpr,'b',label = 'AUC %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plot.xlim([0,1])
    plot.ylim([0,1])
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    #plt.show
    plt.savefig(filepath + modelName + ".png")
    plt.close()

def save_model_results(predicts,modelName,filepath,intervals):
    make_roc_plot(predicts,modelName,filepath)
    path = filepath + modelName + ".txt"
    file1 = open(path,"w+")

    prob = [x / intervals for x in range(1,intervals,1)]

    file1.write("Model performance for model ",modelName)
    file1.write("\n")


    for p in prob:
        pred_labels = np.array([0 if x < p else 1 for x in ynew_prob])

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


def simple_bagging(n,train_features,train_labels,val_features,val_labels,test_features,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE,print_intervals):
    predicts = []
    for i in range(1,n+1):
        print("training number: ",i)
        test_predict = train_single_model(train_features,train_labels,val_features,val_labels,test_features,filepath,modelName,BATCH_SIZE,EPOCHS,PATIENCE)
        predicts.append(test_predict)
    predicts = np.sum(predicts,0)
    predicts = predicts / n
    #print_prob(test_labels,pred_labels,predicts,print_intervals)
    save


