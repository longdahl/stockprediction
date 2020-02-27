import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support as score

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

