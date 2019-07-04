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

datos_preprocesados = False

if not datos_preprocesados:
    #los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
    data_x = pd.read_csv('water_pump_tra.csv')
    data_y = pd.read_csv('water_pump_tra_target.csv')
    data_x_tst = pd.read_csv('water_pump_tst.csv')

    #se quitan las columnas que no se usan
    data_x.drop(labels=['id'], axis=1,inplace = True)
    data_x.drop(labels=['date_recorded'], axis=1,inplace = True)

    data_x_tst.drop(labels=['id'], axis=1,inplace = True)
    data_x_tst.drop(labels=['date_recorded'], axis=1,inplace = True)

    data_y.drop(labels=['id'], axis=1,inplace = True)

    '''
    Seleccion de variables e imputacion
    '''
    print("------ Selección de características...\n")
    var = data_x.shape[1]+1
    # Eliminamos recorded_by ya que todos los valores son el mismo y no aporta nada
    data_x = data_x.drop('recorded_by',axis=1)
    data_x_tst = data_x_tst.drop('recorded_by',axis=1)

    #Eliminamos num_private ya que la mayoria de valores son 0 y no aporta mucho
    data_x = data_x.drop('num_private',axis=1)
    data_x_tst = data_x_tst.drop('num_private',axis=1)

    #Tenemos muchas variables que nos dicen lo mismo, la localizacion, por lo que dejaremos solo 1
    data_x = data_x.drop('subvillage',axis=1)
    data_x_tst = data_x_tst.drop('subvillage',axis=1)
    data_x = data_x.drop('region',axis=1)
    data_x_tst = data_x_tst.drop('region',axis=1)
    data_x = data_x.drop('region_code',axis=1)
    data_x_tst = data_x_tst.drop('region_code',axis=1)
    data_x = data_x.drop('district_code',axis=1)
    data_x_tst = data_x_tst.drop('district_code',axis=1)
    data_x = data_x.drop('ward',axis=1)
    data_x_tst = data_x_tst.drop('ward',axis=1)

    '''
    Se convierten las variables categóricas a variables numéricas (ordinales)
    '''
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    #Train
    strings = [col for col in data_x.columns if data_x[col].dtype=='object']
    df_strings = pd.DataFrame(data_x[strings],index=data_x.index,columns=strings)
    numbers = np.setdiff1d(data_x.columns.values,np.array(strings))
    df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)
    imps = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impn = SimpleImputer(missing_values=0, strategy='mean')
    imps.fit(df_strings)
    impn.fit(df_numbers)
    df_strings = pd.DataFrame(imps.transform(df_strings),index=df_strings.index,columns=df_strings.columns)
    df_numbers = pd.DataFrame(impn.transform(df_numbers),index=df_numbers.index,columns=df_numbers.columns)
    df_numbers['gps_height'] = round(df_numbers['gps_height']).astype(int)
    df_numbers['population'] = round(df_numbers['population']).astype(int)
    df_numbers['construction_year'] = round(df_numbers['construction_year']).astype(int)
    df_numbers['amount_tsh'] = round(df_numbers['amount_tsh']).astype(int)
    df_strings = df_strings.apply(LabelEncoder().fit_transform)

    '''
    df_strings_c = df_strings.copy()
    n = 0
    mod_columns = []
    for col in df_strings.columns:
        if df_strings[col].nunique() < 10:
            aux = pd.get_dummies(df_strings[col])
            index = [str(name) + str(i+n) for i, name in enumerate(aux.columns.values)]
            n += len(index)
            aux.columns = index
            df_strings_c = pd.DataFrame.join(df_strings_c.drop(col, axis=1), aux)
            mod_columns.append(col)
        else:
            aux = df_strings[col].to_frame().apply(LabelEncoder().fit_transform)
            df_strings_c[col] = aux

    df_strings = df_strings_c
    #'''

    #df_numbers['construction_year'] = LabelEncoder().transform(df_numbers['construction_year'])

    data_x_nan = pd.DataFrame.join(df_strings,df_numbers)

    #Test
    strings = [col for col in data_x_tst.columns if data_x_tst[col].dtype=='object']
    df_strings = pd.DataFrame(data_x_tst[strings],index=data_x_tst.index,columns=strings)
    numbers = np.setdiff1d(data_x_tst.columns.values,np.array(strings))
    df_numbers = pd.DataFrame(data_x_tst[numbers],index=data_x_tst.index,columns=numbers)
    df_strings = pd.DataFrame(imps.transform(df_strings),index=df_strings.index,columns=df_strings.columns)
    df_numbers = pd.DataFrame(impn.transform(df_numbers),index=df_numbers.index,columns=df_numbers.columns)
    df_numbers['gps_height'] = round(df_numbers['gps_height']).astype(int)
    df_numbers['population'] = round(df_numbers['population']).astype(int)
    df_numbers['construction_year'] = round(df_numbers['construction_year']).astype(int)
    df_numbers['amount_tsh'] = round(df_numbers['amount_tsh']).astype(int)
    df_strings = df_strings.apply(LabelEncoder().fit_transform)

    '''
    df_strings_c = df_strings.copy()
    n = 0
    not_columns = np.setdiff1d(df_strings_c.columns.values,np.array(mod_columns))
    for col in mod_columns:
        aux = pd.get_dummies(df_strings[col])
        index = [str(name) + str(i+n) for i, name in enumerate(aux.columns.values)]
        n += len(index)
        aux.columns = index
        df_strings_c = pd.DataFrame.join(df_strings_c.drop(col, axis=1), aux)

    for col in not_columns:
        aux = df_strings[col].to_frame().apply(LabelEncoder().fit_transform)
        df_strings_c[col] = aux

    df_strings = df_strings_c
    #'''


    #df_numbers['construction_year'] = LabelEncoder().transform(df_numbers['construction_year'])

    data_x_tst_nan = pd.DataFrame.join(df_strings,df_numbers)
    #'''

    '''
    Ahora que tenemos todos los valores como numeros, usamos la matriz de correlacion para eliminar variables
    '''

    # Eliminamos las variables que tengan un nivel alto de correlacion
    corr_matrix = data_x_nan.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    umbral = 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > umbral) or any(upper[column] < -1.0 * umbral)]
    # to_drop = ['extraction_type_group', 'quantity_group', 'source_type', 'waterpoint_type_group']
    data_x_nan = data_x_nan.drop(to_drop, axis=1)
    data_x_tst_nan = data_x_tst_nan.drop(to_drop, axis=1)
    #'''

    #Guardo el resultado en un fichero
    '''
    data_x_nan.to_csv('data_x_preproces.csv',index=False)
    data_x_tst_nan.to_csv('data_x_tst_preproces.csv',index=False)
    #'''

    print("Antes %d caracteristicas. Ahora %d caracteristicas\n" % (var,data_x_nan.shape[1]))
else:
    data_x_nan = pd.read_csv('data_x_preproces.csv')
    data_y = pd.read_csv('water_pump_tra_target.csv')
    data_y.drop(labels=['id'], axis=1, inplace=True)
    data_x_tst_nan = pd.read_csv('data_x_tst_preproces.csv')


#Desvalanceo de clases

print("------ SMOTE...\n")
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority',n_jobs=4)

X_train , y_train = sm.fit_resample(data_x_nan,data_y.values.ravel())

data_x_nan = pd.DataFrame(X_train,columns=data_x_nan.columns)
data_y = pd.DataFrame(y_train,columns=data_y.columns)
if not datos_preprocesados:
    print("Antes: %d elementos. Ahora %d elementos\n" % (data_x.shape[0],data_x_nan.shape[0]))
#'''

# Guardamos el resultado en listas para el entrenamiento
X = data_x_nan.values
X_tst = data_x_tst_nan.values
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

def index_by_name(y,name):
    index = [i for i in range(len(y)) if y[i] == name]

    return index
#------------------------------------------------------------------------

select_boruta = False

from sklearn.ensemble import RandomForestClassifier
if select_boruta:
    print("------ Selección de características Boruta...")
    from boruta import BorutaPy

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    '''
    Configuramos BorutaPy para la selección de características en función de la configuración hecha
    para Random Forest
    '''
    feat_selector = BorutaPy(rf, max_iter=9, n_estimators=200, verbose=0)

    #Lo aplicamos sobre nuestros datos.
    mask = data_x_nan.isnull()
    data_x_tmp = data_x_nan.fillna(9999)
    X = data_x_tmp.values
    feat_selector.fit(X, y)
    data_x_nan = data_x_tmp.where(~mask, data_x_nan)
    X = data_x_nan.values

    #Comprobar las características (variables) seleccionadas.
    all_features = np.array(list(data_x_nan)[0:])
    selected_features = all_features[feat_selector.support_]

    print("\nCaracterísticas seleccionadas:")
    print(selected_features)

    #Comprobar el ranking de características.
    print("\nRanking de características:")
    print(feat_selector.ranking_)

    #Aplicamos transform() a X para filtrar las características y dejar solo las seleccionadas.
    X_filtered = feat_selector.transform(X)
    X_tst_filtered = feat_selector.transform(X_tst)
    print("\nNúmero de características inicial: {:d}, después de la selección: {:d}\n".format(X.shape[1],X_filtered.shape[1]))

else:
    X_filtered = X
    X_tst_filtered = X_tst

#------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 1000,objective='multi:softmax',n_jobs=8,max_depth=11,num_class=4)
clfb, y_test_clf = validacion_cruzada(clf,X_filtered,y,skf)
#'''

'''
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='multiclass',n_estimators=1000,num_threads=8,max_depth=-1)
lgbmb, y_test_lgbm = validacion_cruzada(lgbm,X_filtered,y,skf)
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

calculate_best = False

if calculate_best:
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV

    params_xgb = {'max_depth': [3, 4, 12, 14],
                  'n_estimators': [100, 200, 500],
                  'num_class': [2, 4, 6],
                  'objective': ['multi:softmax']}

    params_lgbm = {'boosting_type': ['gbdt', 'rf', 'goss', 'dart'],
                  'n_estimators': [1000],
                  'objective': ['multiclass'],
                  'importance_type':['gain']}


    print("------ Grid Search...")
    grid = GridSearchCV(lgbmb, params_lgbm, cv=3, n_jobs=-1, verbose=1, scoring=make_scorer(accuracy_score))
    grid.fit(X_filtered,y)
    print("Mejores parámetros:")
    print(grid.best_params_)
    print("\n------ XGBoosted con los mejores parámetros de GridSearch...")
    gs, y_test_gs = validacion_cruzada(grid.best_estimator_,X_filtered,y,skf)
    #'''

    print("------ Random Search...")
    rndsrch = RandomizedSearchCV(lgbmb, params_lgb, cv=3, n_iter=5, n_jobs=-1, verbose=1, scoring=make_scorer(accuracy_score))
    rndsrch.fit(X_filtered,y)
    print("Mejores parámetros:")
    print(rndsrch.best_params_)
    print("\n------ LightGBM con los mejores parámetros de RandomSearch...")
    gsr, y_test_gsr = validacion_cruzada(rndsrch.best_estimator_,X_filtered,y,skf)
    #'''

'''
Usamos el modelo que mejor nos convenga para guardar los datos finales
'''

clf = clfb.fit(X_filtered,y,feature_name=data_x_nan.columns.tolist(),categorical_feature=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
y_pred_tra = clf.predict(X_filtered)
print("Score clf: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst_clf = clf.predict(X_tst_filtered)
#'''

xgb.plot_importance(clf)

plt.show()

'''
df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst_clf
df_submission.to_csv("submission7lgmb.csv", index=False)
#'''