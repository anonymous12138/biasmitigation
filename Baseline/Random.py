from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd
import random,time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath('..'))
from Measure import *


def blind_random(base_clf, df, keyword, ratio=.2, rep=10):
    dataset_orig = df.dropna()

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    acc, pre, recall, f1 = [], [], [], []
    aod, eod, spd, di, fr = [], [], [], [], []

    for i in range(rep):
        start = time.time()
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=ratio, random_state=i)
        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])
        clf = copy.deepcopy(base_clf)

        X_train_rand = copy.deepcopy(X_train)
        np.random.seed(i)
        thresh = (np.sum(X_train[keyword])/X_train.shape[0])*10
        random_list = np.random.randint(low=0, high=10, size=X_train.shape[0])
        X_train_rand[keyword] = np.where(random_list>thresh,1,0)
        # X_train_rand[keyword] = np.random.randint(low=0, high=1, size=X_train.shape[0])

        clf.fit(X_train_rand, y_train)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        aod.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'aod'))
        eod.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'eod'))
        spd.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'SPD'))
        di.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'DI'))
        flip_rate = calculate_flip(clf, X_test, keyword)
        fr.append(flip_rate)
        print("Round", i + 1, 'finished.')
    res1 = [acc, pre, recall, f1, aod, eod, spd, di, fr]
    return res1



if __name__ == "__main__":
    filenames = ['adult','compas-scores-two-years','bank','default','GermanData','h181','heart']
    keywords = {'adult':['sex','race'],
                'compas-scores-two-years':['sex','race'],
                'bank':['age'],
                'default':['sex'],
                'GermanData':['sex'],
                'h181':['race'],
                'heart':['age']
                }

    # base = LogisticRegression(max_iter=1000)
    base = RandomForestClassifier()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            df = pd.read_csv(fname+"_processed.csv")
            result1 = blind_random(base,df,keyword=keyword,rep=20)
            a, p, r, f, ao, eo, spd, di, fr = result1
            print("**"*50)
            print(fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))
            print("-FR", np.mean(fr))