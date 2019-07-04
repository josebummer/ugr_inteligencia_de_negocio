# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2018
Contenido:
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada

    Algoritmos avanzados en Python
    
    En este notebook vamos a ver un ejemplo de uso de eXtreme Gradient Boosting (XGB), LightGBM y Grid Search en Python.
    
        XGBoost y LightGBM:
            https://www.kaggle.com/nschneider/gbm-vs-xgboost-vs-lightgbm
            https://dnc1994.com/2016/03/installing-xgboost-on-windows/
            https://github.com/Microsoft/LightGBM
            https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide
    
        Grid Search:
            http://scikit-learn.org/stable/modules/grid_search.html
            https://www.kaggle.com/tanitter/grid-search-xgboost-with-scikit-learn?scriptVersionId=23363
    
        Ejemplo en R:
            http://minimaxir.com/2017/02/predicting-arrests/
"""

#Indicamos si queremos que cada variable categórica se convierta en varias binarias (tantas como categorías), indicamos binarizar = True, o si preferimos que cada variable categórica se convierta simplemente a una numérica (ordinal), binarizar = False.
binarizar = False

'''
Leemos el conjunto de datos. Los valores perdidos notados como '?' se convierten a NaN, sino se consideraría '?' como una categoría más.
'''

import pandas as pd
import numpy as np

carpeta_datos="./"
fich_tra = "water_pump_tra"
fich_tst = "water_pump_tst"
fich_tra_y = "water_pump_tra_target"

if not binarizar:
    bank_orig = pd.read_csv(carpeta_datos+fich_tra+'.csv', delimiter=',')
    bank_orig.replace(np.nan,'?')
else:
    bank_orig = pd.read_csv(carpeta_datos+fich_tra+'.csv',na_values="?", delimiter=',')

if not binarizar:
    bank_tst = pd.read_csv(carpeta_datos+fich_tst+'.csv', delimiter=',')
    bank_tst.replace(np.nan, '?')
else:
    bank_tst = pd.read_csv(carpeta_datos+fich_tst+'.csv',na_values="?", delimiter=',')

y_tra = pd.read_csv(carpeta_datos+fich_tra_y+'.csv',delimiter=',')
    
# print("------ Lista de características y tipos (object=categórica)")
# print(bank_orig.dtypes,"\n")
    
'''
Si el dataset contiene variables categóricas con cadenas, es necesario convertirlas a numéricas
antes de usar 'fit'. Si no las vamos a hacer ordinales (binarizar = True), las convertimos a
variables binarias con get_dummies. Para saber más sobre las opciones para tratar variables
categóricas: http://pbpython.com/categorical-encoding.html
'''
    
# devuelve una lista de las características categóricas excluyendo la columna 'class' que contiene la clase
lista_categoricas = [x for x in bank_orig.columns if (bank_orig[x].dtype == object and bank_orig[x].name != 'y')]
binarizar = False
if not binarizar:
    bank = bank_orig
else:
    # reemplaza las categóricas por binarias
    bank = pd.get_dummies(bank_orig, columns=lista_categoricas)
    
#Lista de atributos del dataset.
list(bank)

'''
Separamos el DataFrame en dos arrays numpy, uno con las características (X) y otro con la clase (y).
Si la última columna es la que contiene la clase se puede separar así:
'''    
from sklearn import preprocessing
    
le = preprocessing.LabelEncoder() 
# LabelEncoder codifica los valores originales entre 0 y el número de valores - 1
# Se puede usar para normalizar variables o para transformar variables no-numéricas en numéricas
columns = bank.keys().tolist()
X = bank[columns].values
y = y_tra['status_group'].values
y_bin = le.fit_transform(y)

print("X contiene las características, y contiene la clase asignada:")
print("X:", X)
print("y:", y)

'''
Si las variables categóricas tienen muchas categorías, se generarán muchas variables y algunos
algoritmos serán extremadamente lentos. Se puede optar por, como hemos comentado antes, convertirlas
a variables numéricas (ordinales) sin binarizar.
Esto se haría si no se ha ejecutado pd.get_dummies() previamente. Además, no funciona is hay valores
perdidos notados como NaN.
'''
if not binarizar:    
    for i in range(X.shape[1]):
        if isinstance(X[0,i],str):
            X[:,i] = le.fit_transform(X[:,i])

'''
Validación cruzada con particionado estratificado y control de la aleatoriedad (fijando la semilla).
'''            
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
import numpy

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    y_prob_all = []

    for train, test in cv.split(X, y):
        modelo = modelo.fit(X[train],y[train])
        y_pred = modelo.predict(X[test])
        y_prob = modelo.predict_proba(X[test])[:,1] #la segunda columna es la clase positiva '1' en bank-marketing
        y_test_bin = y[test]
        #y_test_bin = le.fit_transform(y[test]) #se convierte a binario para AUC: 'yes' -> 1 (clase positiva) y 'no' -> 0 en bank-marketing
        
        print("Accuracy: {:6.2f}%, F1-score: {:.4f}, G-mean: {:.4f}, AUC: {:.4f}".format(accuracy_score(y[test],y_pred)*100 , f1_score(y[test],y_pred,average='macro'), geometric_mean_score(y[test],y_pred,average='macro'), roc_auc_score(y_test_bin,y_prob)))
        y_test_all = numpy.concatenate([y_test_all,y_test_bin])
        y_prob_all = numpy.concatenate([y_prob_all,y_prob])

    print("")

    return modelo, y_test_all, y_prob_all

#------------------------------------------------------------------------------
    
import xgboost as xgb

#print("------ XGB...")
#clf = xgb.XGBClassifier(n_estimators = 200)
#clf, y_test_clf, y_prob_clf = validacion_cruzada(clf,X,y,skf)

'''
Light Gradient Boosting
'''
import lightgbm as lgb

print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='multiclass',n_estimators=200,num_threads=4)
lgbm, y_test_lgbm, y_prob_lgbm = validacion_cruzada(lgbm,X,y,skf)

print("------ Importancia de las características...")
importances = list(zip(lgbm.feature_importances_, bank.columns))
importances.sort()
pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind='barh',legend=False)

'''
Selección de características (Feature Selection)
Probablemnte tengáis que instalar el paquete boruta: pip install boruta
Realizamos una selección de características usando Random Forest como estimador.
Configuramos Random Forest
'''
print("------ Selección de características...")
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

'''
Configuramos BorutaPy para la selección de características en función de la configuración hecha
para Random Forest
'''
feat_selector = BorutaPy(rf, max_iter=9, n_estimators=200, verbose=0, random_state=123456)

#Lo aplicamos sobre nuestros datos.
feat_selector.fit(X, y)

#Comprobar las características (variables) seleccionadas.
all_features = numpy.array(list(bank)[0:-1])
selected_features = all_features[feat_selector.support_]

print("\nCaracterísticas seleccionadas:")
print(selected_features)

#Comprobar el ranking de características.
print("\nRanking de características:")
print(feat_selector.ranking_)

#Aplicamos transform() a X para filtrar las características y dejar solo las seleccionadas.
X_filtered = feat_selector.transform(X)
print("\nNúmero de características inicial: {:d}, después de la selección: {:d}\n".format(X.shape[1],X_filtered.shape[1]))

'''
Ejecutamos LGBM sobre el conjunto de datos resultante de la selección de características con validación cruzada estratificada (las mismas particiones que para los demás algoritmos).
'''
print("------ LightGBM sobre las características seleccionadas...")
lgbm, y_test_lgbm, y_prob_lgbm = validacion_cruzada(lgbm,X_filtered,y,skf)

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

params_xgb = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
          'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4], 'n_estimators':[50,100,200]}

print("------ Grid Search...")
params_lgbm = {'feature_fraction':[i/10.0 for i in range(3,6)], 'learning_rate':[0.05,0.1],
               'num_leaves':[30,50], 'n_estimators':[200]}
grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(f1_score))
grid.fit(X,y_bin)
print("Mejores parámetros:")
print(grid.best_params_)
print("\n------ LightGBM con los mejores parámetros de GridSearch...")
gs, y_test_gs, y_prob_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)

print("------ Random Search...")
params_rnd_lgbm = {'feature_fraction':[i/10.0 for i in range(2,7)], 'learning_rate':[i/100.0 for i in range(5,20)],
               'num_leaves':[i*10 for i in range(2,6)], 'n_estimators':[200]}
rndsrch = RandomizedSearchCV(lgbm, params_rnd_lgbm, cv=3, n_iter=5, n_jobs=1, verbose=1, scoring=make_scorer(f1_score))
rndsrch.fit(X,y_bin)
print("Mejores parámetros:")
print(rndsrch.best_params_)
print("\n------ LightGBM con los mejores parámetros de RandomSearch...")
gs, y_test_gs, y_prob_gs = validacion_cruzada(rndsrch.best_estimator_,X,y,skf)


'''
Referencias complementarias

Varios enlaces que pueden ser de utilidad:

    Imputación de valores perdidos: https://pypi.python.org/pypi/fancyimpute
    Desbalanceo de clase: https://github.com/scikit-learn-contrib/imbalanced-learn
    Selección de características con Boruta: https://github.com/scikit-learn-contrib/boruta_py

En Windows para algunos paquetes (por ejemplo, fancyimpute) puede ser necesario instalar previamente esto: http://landinghub.visualstudio.com/visual-cpp-build-tools

En general, para instalación de paquetes: pip install <paquete>
'''



