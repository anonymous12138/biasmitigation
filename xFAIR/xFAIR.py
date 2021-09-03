import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import tree
import sys
sys.path.append(os.path.abspath('..'))
from Measure import *


def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out


def xFAIR(df, base_clf, base2, keyword, ratio=.2, rep=10, smote1=True, verbose=False, thresh=.5):
    dataset_orig = df.dropna()
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    acc, pre, recall, f1 = [], [], [], []
    aod, eod, spd, di = [], [], [], []

    for i in range(rep):
        start = time.time()
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=ratio, random_state=i)
        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        reduced = list(X_train.columns)
        reduced.remove(keyword)
        X_reduced, y_reduced = X_train.loc[:, reduced], X_train[keyword]
        # Build model to predict the protect attribute
        clf1 = copy.deepcopy(base2)
        if smote1:
            sm = SMOTE()
            X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
            clf = copy.deepcopy(base_clf)
            clf.fit(X_trains, y_trains)
            y_proba = clf.predict_proba(X_trains)
            y_proba = [each[1] for each in y_proba]
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_trains, y_trains)
            else:
                clf1.fit(X_trains, y_proba)
        else:
            clf = copy.deepcopy(base_clf)
            clf.fit(X_reduced, y_reduced)
            y_proba = clf.predict_proba(X_reduced)
            y_proba = [each[1] for each in y_proba]
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_reduced, y_reduced)
            else:
                clf1.fit(X_reduced, y_proba)
        #             clf1.fit(X_reduced,y_reduced)

        if verbose:
            if isinstance(clf1, LinearRegression):
                importances = (clf1.coef_)
            elif isinstance(clf1, LogisticRegression):
                importances = (clf1.coef_[0])
                print("coef:", clf1.coef_[0], "intercept:", clf1.intercept_)
            else:
                importances = clf1.feature_importances_
            indices = np.argsort(importances)
            features = X_reduced.columns

            plt.rcParams.update({'font.size': 14})
            plt.title('Feature Importances on sensitive attribute')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()

        X_test_reduced = X_test.loc[:, X_test.columns != keyword]
        protected_pred = clf1.predict(X_test_reduced)
        if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
            protected_pred = reg2clf(protected_pred, threshold=thresh)
        # Build model to predict the taget attribute Y
        clf2 = copy.deepcopy(base_clf)

        X_test.loc[:, keyword] = protected_pred
        y_pred = clf2.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        aod.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'aod'))
        eod.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'eod'))
        spd.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'SPD'))
        di.append(measure_final_score(dataset_orig_test, y_pred, cm, X_train, y_train, X_test, y_test, keyword, 'DI'))
        print("Round", (i + 1), "finished.")
        print('Time', time.time() - start)
    res1 = [acc, pre, recall, f1, aod, eod, spd, di]
    return res1


if __name__ == "__main__":
    filenames = ['adult', 'compas-scores-two-years', 'GermanData', 'bank', 'heart', 'default', 'h181']
    keywords = {'adult': ['sex', 'race'],
                'compas-scores-two-years': ['sex', 'race'],
                'bank': ['age'],
                'default': ['sex'],
                'GermanData': ['sex'],
                'h181': ['race'],
                'heart': ['age']
                }
    base = RandomForestClassifier()
    base2 = DecisionTreeRegressor()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            df1 = pd.read_csv("./Data/"+fname + "_processed.csv")
            result1 = xFAIR(df1, base,base2, keyword=keyword, rep=20,verbose=True)
            a, p, r, f, ao, eo, spd, di = result1
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