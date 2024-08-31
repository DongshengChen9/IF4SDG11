import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
import pickle

num_repeat = 10

cities_mean_kappa =[]
cities_accuracy =[]
with open('./input/result_coarse.pkl', 'rb') as f:
    data = pickle.load(f)
    kappa_scores = []
    accuracy_scores = []
    X = data['feats']
    y = data['label'][:X.shape[0]]
    rf_model = RandomForestClassifier(bootstrap=False,n_estimators=200,random_state=1)
    for ite in range(num_repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=ite)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        overall_test = np.append(y_test,data['result_outcoarse'])
        overall_pred = np.append(y_pred,data['label'][X.shape[0]:])
        kappa = cohen_kappa_score(overall_test, overall_pred)
        kappa_scores.append(kappa)
        accuracy = accuracy_score(overall_test, overall_pred)
        accuracy_scores.append(accuracy)
    print(f"Kappa\tnp.mean(kappa_scores)\tnp.mean(accuracy_scores)")