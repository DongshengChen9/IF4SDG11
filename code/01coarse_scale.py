import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

def calculate_average_feats(object_layer, pixel_layer):
    polygons = gpd.read_file(object_layer)
    points = gpd.read_file(pixel_layer) 
    joined = gpd.sjoin(points, polygons, how='inner', op='within')
    grouped = joined.groupby('FID_left').mean()
    polygons = polygons.drop(columns=[f'feat_{i}' for i in range(1, 49)])
    polygons = polygons.merge(grouped, left_on='FID', right_on='FID_left')
    return polygons


## prepare your own results of object segmentation and pixels with vgg features and place them in input folder
objects_shp = './input/objects_with_spectral_textual.shp'
pixels_shp = './input/pixels_with_vgg_poi.shp'
objects = calculate_average_feats(objects_shp, pixels_shp)

labelled_objects_shp = './input/labelled_objects_with_spectral_textual.shp'
labelled_pixels_shp = './input/labelled_pixels_with_vgg_poi.shp'

labelled_output_shp = calculate_average_feats(labelled_objects_shp, labelled_pixels_shp)

features = labelled_output_shp[[f'feat_{i}' for i in range(1, 552)]]
labels = labelled_output_shp['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.1, random_state=1)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Coarse scale accuracy:", accuracy_score(y_test, y_pred))

features = objects[[f'feat_{i}' for i in range(1, 552)]]
scaler = StandardScaler()
X = scaler.fit_transform(features)
y_pred = rf_model.predict(X)
objects['pred'] = y_pred
classified_objects = objects[objects['pred'] != 0]
points = gpd.read_file(pixels_shp)
samples_fine = points.overlay(classified_objects, how='intersection')
samples_outcoarse = points.overlay(classified_objects, how='difference')
dict_data = {}
dict_data['feats'] = scaler.fit_transform(samples_fine[[f'feat_{i}' for i in range(1, 552)]])
dict_data['label'] = np.append(samples_fine['label'].tolist(),samples_outcoarse['label'].tolist())
dict_data['result_outcoarse'] = samples_outcoarse.shape[0] * [0]
with open('./input/result_coarse.pkl', 'wb') as f:
    pickle.dump(dict_data, f)   
