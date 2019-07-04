"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2018
Contenido:
    Uso simple de XGB y LightGBM para competir en DrivenData:
       https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import time

from sklearn.datasets import samples_generator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

'''
lectura de datos
'''

data_x_nan = pd.read_csv('data_x_preproces.csv')
data_y = pd.read_csv('water_pump_tra_target.csv')
data_y.drop(labels=['id'], axis=1, inplace=True)
data_x_tst_nan = pd.read_csv('data_x_tst_preproces.csv')


# Guardamos el resultado en listas para el entrenamiento
X_filtered = data_x_nan.values
X_tst_filtered = data_x_tst_nan.values
y = np.ravel(data_y.values)


#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv, name=None):
    y_test_all = []

    if name!=None:
        f = open('salidas/'+name+'.csv','w')
        i = 0
        f.write('it,acc,tiempo\n')
    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        if name != None:
            f.write('%d,%0.4f,%6.2f\n' % (i,accuracy_score(y[test],y_pred),tiempo))
            i += 1
        y_test_all = np.concatenate([y_test_all,y[test]])

    if name != None:
        f.close()
    print("")

    return modelo, y_test_all

#------------------------------------------------------------------------


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

'''
print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 500,objective='multi:softmax',n_jobs=8,max_depth=11,num_class=4)
clfb, y_test_clf = validacion_cruzada(clf,X_filtered,y,skf)
#'''


print("------ LightGBM...")
lgbmb = lgb.LGBMClassifier(objective='multiclass',n_estimators=1000,num_threads=8,max_depth=-1)
#lgbmb, y_test_lgbm = validacion_cruzada(lgbm,X_filtered,y,skf,'salida7lgb')
#'''

'''
print("------ MLPNN...")
X_filtered = preprocessing.normalize(X_filtered)
nn = MLPClassifier()
nnb, y_test_nn = validacion_cruzada(nn,X_filtered,y,skf,'salida5nn')
#'''

'''
print("------ SVC...")
svc = SVC()
svcb, y_test_svc = validacion_cruzada(svc,X_filtered,y,skf,'salida5svc')
#'''

'''
print("------ RandomForest...")
rf = RandomForestClassifier(n_estimators=1000,min_samples_split=10,oob_score=True,n_jobs=-1)
rfb, y_test_svc = validacion_cruzada(rf,X_filtered,y,skf)
#'''



'''
Usamos el modelo que mejor nos convenga para guardar los datos finales
'''
columnas = data_x_nan.columns.values
columnas = columnas[0:21].tolist()

lgbm = lgbmb.fit(X_filtered,y,feature_name=data_x_nan.columns.tolist(),categorical_feature=columnas)
y_pred_tra = lgbm.predict(X_filtered)
print("Score clf: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst_clf = lgbm.predict(X_tst_filtered)
#'''

lgb.plot_importance(lgbm)

plt.show()


df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst_clf
df_submission.to_csv("submission7lgmb.csv", index=False)
#'''