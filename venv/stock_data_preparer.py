import pandas as pd
import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import class_weight

#TODO change format of stock data


def pickle_save(to_save,save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(to_save, fp)


def pickle_load(load_path):

    with open(load_path, "rb") as fp:
        b = pickle.load(fp)
    return b

def createPCA(label,features, save_path, max_comp):
    features = standardize(features)
    pca = PCA(n_components=features.shape[1])

    pca.fit(features)

    var_expl = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)

    # get number of components that explains 95% of the variance or a maximum of max_comp
    comp = min(next(x[0] for x in enumerate(var_expl) if x[1] > 95), max_comp)

    if comp == max_comp:
        print("the number of principal components are ", comp, "with an explained variance of: ", var_expl[max_comp])
    else:
        print("the number of principal components are: ", comp)

    pca = PCA(n_components=comp)

    principalComponents = pca.fit_transform(features)
    principalComponents = pd.concat([label,principalComponents], axis=1, ignore_index=True)
    column_names = ["result"] + ["PCA var " + str(x) for x in range(1, comp + 1)]
    principalComponents = pd.DataFrame(data=principalComponents, columns=column_names)
    pickle_save(var_expl, save_path + "PCA_var_explained_sp500.txt")
    principalComponents.to_pickle(save_path + "PCA_sp500.txt")


def binary_upsampling(train):

    num_0 = len(train[train.iloc[:,0] == 0])
    num_1 = len(train[train.iloc[:, 0] == 1])

    if num_0 > num_1:
        print("upsampling number of ones")
        samples = num0

    elif num_1 < num_0:
        print("upsampling number of ones")
        samples = num0
    else:
        print("train_features = np.array(train)No upsampling required")

    upsampled = resample(replace = True,n_samples = samples,random_state = 27)
    return upsampled

def date_preparer(data):

    weekday = [calendar.day_name[x.weekday()] for x in data['date']]

    month = [calendar.month_name[x.month] for x in data['date']]

    year = [x.year for x in data['date']]

    year_dummies = pd.get_dummies(year, drop_first=True)
    month_dummies = pd.get_dummies(month, drop_first=True)
    weekday_dummies = pd.get_dummies(weekday, drop_first=True)

    earliest_date = data['date'].min()

    year_dummies.columns = ["year_dummy_" + str(x) for x in range(earliest_date.year + 1, earliest_date.year + 1 + year_dummies.shape[1])]

    elapsed_days = pd.DataFrame([(x - earliest_date).days for x in data['date']])
    elapsed_days.columns = ["Elapsed_days"]
    data = data.drop(['date'],axis = 1)

    data_new = pd.concat([data,year_dummies,month_dummies,weekday_dummies,elapsed_days],axis = 1,ignore_index = False)


    return data_new



save_path = "C:/Users/Mikkel/Desktop/machine learning/stock_prediction/"
#save_path_work  =


data = pd.read_pickle(save_path + "sp_500_data_2019okt2020feb18.txt")

data = date_preparer(data)


data.to_pickle(save_path + "sp_500_data_2019okt2020feb18_ver2.txt")

