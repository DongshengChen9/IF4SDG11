import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
import pickle

num_repeat = 10

cities_mean_kappa =[]
cities_accuracy =[]

"""Note that the data file should be put under the directory ./input/"""
data_file = f'./input/example_data.pkl'
if not os.path.exists(data_file):
    print("Please download input data from https://figshare.com/articles/dataset/Example_data_Large-scale_geographic_and_demographic_characterisation_of_informal_settlements_fusing_remote_sensing_POI_and_open_geo-data_/26887177")
    exit()


with open(data_file, 'rb') as f:
    data = pickle.load(f)
for _index in data.keys():
    kappa_scores = []
    accuracy_scores = []
    X = data[_index]['feats']
    y = data[_index]['label'][:X.shape[0]]
    rf_model = RandomForestClassifier(bootstrap=False,n_estimators=200,random_state=1)
    for ite in range(num_repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=ite)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        overall_test = np.append(y_test,data[_index]['result_outcoarse'])
        overall_pred = np.append(y_pred,data[_index]['label'][X.shape[0]:])
        kappa = cohen_kappa_score(overall_test, overall_pred)
        kappa_scores.append(kappa)
        accuracy = accuracy_score(overall_test, overall_pred)
        accuracy_scores.append(accuracy)
    cities_mean_kappa.append(np.mean(kappa_scores))
    cities_accuracy.append(np.mean(accuracy_scores))
    
print(f"Repeated {num_repeat} times\tKappa\tAccuracy")
print(f"Guangzhou\t{round(cities_mean_kappa[2],3)}\t{round(cities_accuracy[2],4)}")
print(f"Shenzhen\t{round(cities_mean_kappa[5],3)}\t{round(cities_accuracy[5],4)}")
print(f"Zhuhai\t\t{round(cities_mean_kappa[6],3)}\t{round(cities_accuracy[6],4)}")
print(f"Foshan\t\t{round(cities_mean_kappa[1],3)}\t{round(cities_accuracy[1],4)}")
print(f"Jiangmen\t{round(cities_mean_kappa[4],3)}\t{round(cities_accuracy[4],4)}")
print(f"Dongguan\t{round(cities_mean_kappa[0],3)}\t{round(cities_accuracy[0],4)}")
print(f"Zhongshan\t{round(cities_mean_kappa[8],3)}\t{round(cities_accuracy[8],4)}")
print(f"Huizhou\t\t{round(cities_mean_kappa[3],3)}\t{round(cities_accuracy[3],4)}")
print(f"Zhaoqing\t{round(cities_mean_kappa[7],3)}\t{round(cities_accuracy[7],4)}")
print(f"Mean\t\t{round(np.mean(cities_mean_kappa),3)}\t{round(np.mean(cities_accuracy),4)}")