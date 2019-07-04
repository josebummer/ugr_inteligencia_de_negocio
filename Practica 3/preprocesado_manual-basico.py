#Leemos los datos
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from collections import defaultdict
import xgboost as xgb
import matplotlib.pyplot as plt

def missing_values(data_x_p,impnu=None,impst=None):
    data_x = data_x_p.copy()
    if impnu != None:
        strings = [col for col in data_x.columns if data_x[col].dtype == 'object']
        df_strings = pd.DataFrame(data_x[strings], index=data_x.index, columns=strings)
        numbers = np.setdiff1d(data_x.columns.values, np.array(strings))
        df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)
        df_strings = pd.DataFrame(impst.transform(df_strings), index=df_strings.index, columns=df_strings.columns)
        df_numbers = pd.DataFrame(impnu.transform(df_numbers), index=df_numbers.index, columns=df_numbers.columns)

        data_x_r = pd.DataFrame.join(df_strings, df_numbers)

        return data_x_r, impnu, impst
    else:
        strings = [col for col in data_x.columns if data_x[col].dtype == 'object']
        df_strings = pd.DataFrame(data_x[strings], index=data_x.index, columns=strings)
        numbers = np.setdiff1d(data_x.columns.values, np.array(strings))
        df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)
        imps = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        impn = SimpleImputer(missing_values=0, strategy='mean')
        imps.fit(df_strings)
        impn.fit(df_numbers)
        df_strings = pd.DataFrame(imps.transform(df_strings), index=df_strings.index, columns=df_strings.columns)
        df_numbers = pd.DataFrame(impn.transform(df_numbers), index=df_numbers.index, columns=df_numbers.columns)

        data_x_r = pd.DataFrame.join(df_strings, df_numbers)

        return data_x_r,impn,imps

def categoricas_ordinales(data_x_p,d):
    data_x = data_x_p.copy()

    strings = [col for col in data_x.columns if data_x[col].dtype == 'object']
    df_strings = pd.DataFrame(data_x[strings], index=data_x.index, columns=strings)
    numbers = np.setdiff1d(data_x.columns.values, np.array(strings))
    df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)

    no_element = []
    for key,la in d.items():
        n_element = 0
        col = df_strings[key]
        values_col = col.unique()
        for elem in values_col:
            esta = any(la.classes_ == elem)
            if not esta:
                index = df_strings.index[df_strings[key]==elem]
                df_strings.at[index,key] = 9999
                n_element += 1
        no_element.append(n_element)


    mask = df_strings.isna()  # mÃ¡scara para luego recuperar los NaN
    data_x_tmp = df_strings.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado
    data_x_tmp = data_x_tmp.astype(str).apply(lambda x: d[x.name].transform(x))  # se convierten categÃ³ricas en numÃ©ricas
    df_strings = data_x_tmp.where(~mask, df_strings)  # se recuperan los NaN

    data_x = pd.DataFrame.join(df_strings, df_numbers)

    return data_x

def categoricas_dummy(data_x_p,data_x_tst_p,delete=True):
    data_x = data_x_p.copy()
    data_x_tst = data_x_tst_p.copy()
    # Train
    strings = [col for col in data_x.columns if data_x[col].dtype == 'object']
    df_strings = pd.DataFrame(data_x[strings], index=data_x.index, columns=strings)
    numbers = np.setdiff1d(data_x.columns.values, np.array(strings))
    df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)

    if delete:
        drop_columns = []
        for col in df_strings:
            if df_strings[col].isna().sum() > 0:
                df_strings = df_strings.drop(col, axis=1)
                drop_columns.append(col)
        # '''

    df_strings_c = df_strings.copy()
    n = 0
    mod_columns = []
    for col in df_strings.columns:
        if df_strings[col].nunique() < 10:
            aux = pd.get_dummies(df_strings[col])
            index = [str(name) + str(i + n) for i, name in enumerate(aux.columns.values)]
            n += len(index)
            aux.columns = index
            df_strings_c = pd.DataFrame.join(df_strings_c.drop(col, axis=1), aux)
            mod_columns.append(col)
        else:
            mask = df_strings[col].to_frame().isnull()
            data_x_tmp = df_strings[col].to_frame().fillna(9999)
            data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
            aux = data_x_tmp.where(~mask, data_x_tmp)
            df_strings_c[col] = aux.copy()

    df_strings = df_strings_c.copy()
    # '''

    data_x = pd.DataFrame.join(df_strings, df_numbers)

    # Test
    strings = [col for col in data_x_tst.columns if data_x_tst[col].dtype == 'object']
    df_strings = pd.DataFrame(data_x_tst[strings], index=data_x_tst.index, columns=strings)
    numbers = np.setdiff1d(data_x_tst.columns.values, np.array(strings))
    df_numbers = pd.DataFrame(data_x_tst[numbers], index=data_x_tst.index, columns=numbers)

    if delete:
        for col in drop_columns:
            df_strings = df_strings.drop(col, axis=1)

    df_strings_c = df_strings.copy()
    n = 0
    not_columns = np.setdiff1d(df_strings_c.columns.values, np.array(mod_columns))
    for col in mod_columns:
        aux = pd.get_dummies(df_strings[col])
        index = [str(name) + str(i + n) for i, name in enumerate(aux.columns.values)]
        n += len(index)
        aux.columns = index
        df_strings_c = pd.DataFrame.join(df_strings_c.drop(col, axis=1), aux)

    for col in not_columns:
        mask = df_strings[col].to_frame().isnull()
        data_x_tmp = df_strings[col].to_frame().fillna(9999)
        data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
        aux = data_x_tmp.where(~mask, data_x_tmp)
        df_strings_c[col] = aux.copy()

    df_strings = df_strings_c.copy()

    data_x_tst = pd.DataFrame.join(df_strings, df_numbers)

    return data_x,data_x_tst

def create_labelEncoder(data_x_p,columns_p):
    data_x = data_x_p.copy()
    columns = columns_p.copy()
    columns.pop(-1)
    data_x = data_x[columns]
    d = defaultdict(LabelEncoder)

    strings = [col for col in data_x.columns if data_x[col].dtype == 'object']
    df_strings = pd.DataFrame(data_x[strings], index=data_x.index, columns=strings)
    numbers = np.setdiff1d(data_x.columns.values, np.array(strings))

    data_x_tmp = df_strings.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado
    nulls = np.full(fill_value='9999', shape=(1,len(df_strings.columns.values)))
    nulls = pd.DataFrame(nulls,columns=df_strings.columns)
    data_x_tmp = data_x_tmp.append(nulls,ignore_index=True)
    data_x_tmp = data_x_tmp.astype(str).apply(lambda x: d[x.name].fit(x))  # se convierten categÃ³ricas en numÃ©ricas

    return d


def knn_imputation(data_x_p,data_x_tst_p,columns_p,col,K=5,clasificacion=False):
    data_x = data_x_p.copy()
    data_x_tst = data_x_tst_p.copy()
    columns = columns_p.copy()
    # TRAIN

    #COnjunto de entrenamiento
    if clasificacion:
        nulos = data_x.isna()
        index_no_null = nulos.index[nulos[col]==False]
        data_x_n = data_x.loc[index_no_null]
    else:
        data_x_n = data_x.loc[data_x[col] != 0]
    columns.append(col)
    data_x_n = data_x_n[columns]

    '''
    aux = data_x_n.isna()
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_n = data_x_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)
    #'''
    '''
    aux = data_x_n == 0
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_n = data_x_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)
    #'''

    data_y_n = data_x_n[col].values.ravel()
    data_x_n = data_x_n.drop(col, axis=1)

    '''
    strings = [col for col in data_x_n.columns if data_x_n[col].dtype == 'object']
    df_strings = pd.DataFrame(data_x_n[strings], index=data_x_n.index, columns=strings)
    numbers = np.setdiff1d(data_x_n.columns.values, np.array(strings))
    df_numbers = pd.DataFrame(data_x_n[numbers], index=data_x_n.index, columns=numbers)
    imps = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impn = SimpleImputer(missing_values=0, strategy='mean')
    imps.fit(df_strings)
    impn.fit(df_numbers)
    #'''

    la = create_labelEncoder(data_x,columns)

    data_x_n = data_x_n.fillna(9999)
    data_x_n = categoricas_ordinales(data_x_n,la)

    norm = MinMaxScaler()
    norm.fit(data_x_n)
    data_x_n = pd.DataFrame(norm.transform(data_x_n), index=data_x_n.index, columns=data_x_n.columns)

    # Calculamos los nulos porque los necesitamos
    if clasificacion:
        nulos = data_x.isna()
        index_null = nulos.index[nulos[col] == True]
        nulos = data_x.loc[index_null]
    else:
        nulos = data_x.loc[data_x[col] == 0]
    nulos = nulos[columns]
    nulos = nulos.drop(col, axis=1)

    #Imputamos los nulos

    #nulos,impn,imps = missing_values(nulos,impnu=impn,impst=imps)
    nulos = nulos.fillna(9999)

    nulos = categoricas_ordinales(nulos,la)

    if clasificacion:
        neight = KNeighborsClassifier(n_neighbors=K, n_jobs=-1,weights='distance',algorithm='brute')
        neight.fit(data_x_n, data_y_n)
    else:
        neight = KNeighborsRegressor(n_neighbors=K, n_jobs=-1,weights='distance',algorithm='brute')
        neight.fit(data_x_n, data_y_n)

    nulos = pd.DataFrame(norm.transform(nulos), index=nulos.index, columns=nulos.columns)

    labels = neight.predict(nulos)

    data_x.iloc[nulos.index, data_x.columns.get_loc(col)] = labels

    # Test
    '''
    if clasificacion:
        nulos = data_x_tst.isna()
        index_no_null = nulos.index[nulos[col]==False]
        data_x_tst_n = data_x_tst.loc[index_no_null]
    else:
        data_x_tst_n = data_x_tst.loc[data_x_tst[col] > 0]
    data_x_tst_n = data_x_tst_n[columns]

    aux = data_x_tst_n.isna()
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_tst_n = data_x_tst_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)

    aux = data_x_tst_n == 0
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_tst_n = data_x_tst_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)

    data_y_n = data_x_tst_n[col].values.ravel()
    data_x_tst_n = data_x_tst_n.drop(col, axis=1)

    #data_x_tst_n = pd.DataFrame(MinMaxScaler().fit_transform(data_x_tst_n), index=data_x_tst_n.index,columns=data_x_tst_n.columns)

    for colum in data_x_tst_n.columns:
        if (data_x_tst_n[colum].max() - data_x_tst_n[colum].min()) == 0:
            data_x_tst_n[colum] = 0.0
        else:
            data_x_tst_n[colum] = (data_x_tst_n[colum] - data_x_tst_n[colum].min()) / (data_x_tst_n[colum].max() - data_x_tst_n[colum].min())

    #'''
    if clasificacion:
        nulos = data_x_tst.isna()
        index_null = nulos.index[nulos[col] == True]
        nulos = data_x_tst.loc[index_null]
    else:
        nulos = data_x_tst.loc[data_x_tst[col] == 0]
    nulos = nulos[columns]
    nulos = nulos.drop(col, axis=1)

    '''
    if clasificacion:
        neight = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_tst_n, data_y_n.astype(int))
    else:
        neight = KNeighborsRegressor(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_tst_n, data_y_n)
    #'''

    if len(nulos > 0):

        #nulos,impn,imps = missing_values(nulos,impnu=impn,impst=imps)
        nulos = nulos.fillna(9999)

        nulos = categoricas_ordinales(nulos,la)

        nulos = pd.DataFrame(norm.transform(nulos), index=nulos.index, columns=nulos.columns)

        labels = neight.predict(nulos)

        data_x_tst.iloc[nulos.index, data_x_tst.columns.get_loc(col)] = labels

    return data_x,data_x_tst

def validacion_cruzada(modelo, X, y, cv, name=None):
    y_test_all = []

    if name!=None:
        f = open('salidas/'+name+'.csv','w')
        i = 0
        f.write('it,acc,tiempo\n')
    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X.iloc[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X.iloc[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        if name != None:
            f.write('%d,%0.4f,%6.2f\n' % (i,accuracy_score(y[test],y_pred),tiempo))
            i += 1
        y_test_all = np.concatenate([y_test_all,y[test]])

    if name != None:
        f.close()
    print("")

    return modelo, y_test_all

def validacion_cruzada_doble(modelo1,modelo2, X, y, cv, name=None):
    y_test_all = []

    if name!=None:
        f = open('salidas/'+name+'.csv','w')
        i = 0
        f.write('it,acc,tiempo\n')
    for train, test in cv.split(X, y):
        t = time.time()
        modelo1 = modelo1.fit(X.iloc[train],y[train])
        modelo2 = modelo2.fit(X.iloc[train], y[train])
        tiempo = time.time() - t
        y_pred1_prob = modelo1.predict_proba(X.iloc[test])
        y_pred2_prob = modelo2.predict_proba(X.iloc[test])

        d = {0:'functional',1:'functional needs repair',2:'non functional'}

        y_pred_tst = []
        for elem1,elem2 in zip(y_pred1_prob,y_pred2_prob):
                valores = elem1+elem2
                y_pred_tst.append(d[valores.argmax()])

        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred_tst) , tiempo))
        if name != None:
            f.write('%d,%0.4f,%6.2f\n' % (i,accuracy_score(y[test],y_pred_tst),tiempo))
            i += 1
        y_test_all = np.concatenate([y_test_all,y[test]])

    if name != None:
        f.close()
    print("")

    return modelo1,modelo2, y_test_all

def validacion_cruzada_triple(modelo1,modelo2,modelo3, X, y, cv, name=None):
    y_test_all = []

    if name!=None:
        f = open('salidas/'+name+'.csv','w')
        i = 0
        f.write('it,acc,tiempo\n')
    for train, test in cv.split(X, y):
        t = time.time()
        modelo1 = modelo1.fit(X.iloc[train],y[train])
        modelo2 = modelo2.fit(X.iloc[train], y[train])
        modelo3 = modelo3.fit(X.iloc[train], y[train])
        tiempo = time.time() - t
        y_pred1_prob = modelo1.predict_proba(X.iloc[test])
        y_pred2_prob = modelo2.predict_proba(X.iloc[test])
        y_pred3_prob = modelo3.predict_proba(X.iloc[test])

        d = {0:'functional',1:'functional needs repair',2:'non functional'}

        y_pred_tst = []
        for elem1,elem2,elem3 in zip(y_pred1_prob,y_pred2_prob,y_pred3_prob):
                valores = elem1+elem2+elem3
                y_pred_tst.append(d[valores.argmax()])

        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred_tst) , tiempo))
        if name != None:
            f.write('%d,%0.4f,%6.2f\n' % (i,accuracy_score(y[test],y_pred_tst),tiempo))
            i += 1
        y_test_all = np.concatenate([y_test_all,y[test]])

    if name != None:
        f.close()
    print("")

    return modelo1,modelo2,modelo3, y_test_all

def prediccion_conjunta_doble(y_pred1,y_pred2):
    d = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}

    y_pred_tst = []
    for elem1, elem2 in zip(y_pred1, y_pred2):
            valores = elem1 + elem2
            y_pred_tst.append(d[valores.argmax()])

    return y_pred_tst

def prediccion_conjunta_triple(y_pred1,y_pred2,y_pred3):
    d = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}

    y_pred_tst = []
    for elem1, elem2, elem3 in zip(y_pred1, y_pred2, y_pred3):
            valores = elem1 + elem2 + elem3
            y_pred_tst.append(d[valores.argmax()])

    return y_pred_tst

#'''

'''
Leemos los datos
'''
print("------ Lectura de datos...")
data_x = pd.read_csv('water_pump_tra.csv')
data_x_tst = pd.read_csv('water_pump_tst.csv')
data_y = pd.read_csv('water_pump_tra_target.csv')
data_y.drop(labels=['id'], axis=1, inplace=True)
data_x = data_x.drop('id',axis=1)
data_x_tst = data_x_tst.drop('id',axis=1)

'''
Modificamos la variable de fecha
'''
#Train
date = pd.to_datetime(data_x['date_recorded'])
mes = date.apply(lambda x: x.month)
maximo = date.max()
ordinal = date.apply(lambda x: (maximo-x).days)

data_x['mes_recorded'] = mes
data_x['date_recorded'] = ordinal

#Test
date = pd.to_datetime(data_x_tst['date_recorded'])
mes = date.apply(lambda x: x.month)
ordinal = date.apply(lambda x: (maximo-x).days)

data_x_tst['mes_recorded'] = mes
data_x_tst['date_recorded'] = ordinal

'''
Imputacion
'''
datos_cargados = False

if not datos_cargados:

    print("------ Imputacion...")

    # subvillage
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
    #                                    ['longitude', 'latitude', 'ward', 'lga',
    #                                     'district_code', 'region_code', 'basin'], 'subvillage', clasificacion=True)

    #funder
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
      #                                  ['installer','ward', 'scheme_management', 'lga', 'district_code','waterpoint_type','quality_group','water_quality',
      #                                   'region_code', 'subvillage', 'basin','management','management_group','waterpoint_type_group','quantity'], 'funder', clasificacion=True)

    #installer
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
    #                                    ['funder','ward', 'scheme_management', 'lga', 'district_code','waterpoint_type','quality_group','water_quality',
     #                                    'region_code', 'subvillage', 'basin','management','management_group','waterpoint_type_group','quantity'], 'installer', clasificacion=True)


    #longitude y latitude
    i_ceros = data_x.index[data_x['latitude']==-0.00000002]
    data_x.at[i_ceros, 'latitude'] = 0.0
    i_ceros = data_x_tst.index[data_x_tst['latitude'] == -0.00000002]
    data_x_tst.at[i_ceros, 'latitude'] = 0.0
    data_x,data_x_tst = knn_imputation(data_x,data_x_tst,['basin','subvillage','region','district_code','lga','ward'],'longitude')
    data_x,data_x_tst = knn_imputation(data_x,data_x_tst,['basin','subvillage','region','district_code','lga','ward'],'latitude')


    #Gps_height
    data_x,data_x_tst = knn_imputation(data_x,data_x_tst,['basin','subvillage','region','district_code','lga','ward','longitude','latitude','extraction_type','waterpoint_type','waterpoint_type_group'],'gps_height')

    # scheme_name
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
    #                                    ['scheme_management', 'funder', 'installer', 'lga','date_recorded',
    #                                     'district_code', 'region_code', 'basin','subvillage','latitude','longitude','ward'], 'scheme_name', clasificacion=True)

    # wpt_name
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
    #                                    ['funder', 'installer', 'date_recorded', 'mes_recorded', 'ward',
    #                                     'scheme_management', 'lga', 'district_code',
    #                                     'region_code', 'subvillage', 'basin','longitude','latitude','gps_height'], 'wpt_name', clasificacion=True)

    # construction_year
    #data_x, data_x_tst = knn_imputation(data_x, data_x_tst,
    #                                    ['funder', 'installer', 'longitude', 'latitude','gps_height', 'lga', 'ward', 'district_code','date_recorded','mes_recorded',
     #                                    'region_code', 'subvillage', 'basin', 'waterpoint_type', 'waterpoint_type_group','quantity','water_quality'],
      #                                  'construction_year')

    #population
    #data_x,data_x_tst = knn_imputation(data_x,data_x_tst,['date_recorded','basin','subvillage','region','district_code','lga','ward','longitude','latitude',
     #                                                     'extraction_type','payment_type','water_quality','quantity'],'population')


    #amount_tsh
    #data_x,data_x_tst = knn_imputation(data_x,data_x_tst,['longitude','latitude','basin','subvillage','region_code','waterpoint_type','construction_year',
     #                                                    'lga','ward','water_quality','quality_group','quantity','quantity_group','extraction_type','payment','source','date_recorded','mes_recorded'],'amount_tsh')


    data_x.to_csv('data_x_preproces.csv',index=False)
    data_x_tst.to_csv('data_x_tst_preproces.csv',index=False)

else:
    data_x = pd.read_csv('data_x_preproces.csv')
    data_x_tst = pd.read_csv('data_x_tst_preproces.csv')
    #'''

'''
Eliminacion de variables
'''


print("------ Eliminacion de variables por inspeccion ocular...")

data_x = data_x.drop('recorded_by',axis=1)
data_x_tst = data_x_tst.drop('recorded_by',axis=1)

data_x = data_x.drop('num_private',axis=1)
data_x_tst = data_x_tst.drop('num_private',axis=1)

data_x = data_x.drop('public_meeting',axis=1)
data_x_tst = data_x_tst.drop('public_meeting',axis=1)

data_x = data_x.drop('permit',axis=1)
data_x_tst = data_x_tst.drop('permit',axis=1)

data_x = data_x.drop('extraction_type_group',axis=1)
data_x_tst = data_x_tst.drop('extraction_type_group',axis=1)

data_x = data_x.drop('payment_type',axis=1)
data_x_tst = data_x_tst.drop('payment_type',axis=1)

data_x = data_x.drop('water_quality',axis=1)
data_x_tst = data_x_tst.drop('water_quality',axis=1)

data_x = data_x.drop('quantity_group',axis=1)
data_x_tst = data_x_tst.drop('quantity_group',axis=1)

data_x = data_x.drop('source_type',axis=1)
data_x_tst = data_x_tst.drop('source_type',axis=1)

data_x = data_x.drop('waterpoint_type_group',axis=1)
data_x_tst = data_x_tst.drop('waterpoint_type_group',axis=1)

data_x = data_x.drop('region',axis=1)
data_x_tst = data_x_tst.drop('region',axis=1)


#data_x = data_x.drop('wpt_name',axis=1)
#data_x_tst = data_x_tst.drop('wpt_name',axis=1)

#data_x = data_x.drop('subvillage',axis=1)
#data_x_tst = data_x_tst.drop('subvillage',axis=1)

#data_x = data_x.drop('scheme_name',axis=1)
#data_x_tst = data_x_tst.drop('scheme_name',axis=1)
#'''

'''
eliminar
'''
'''
#scheme_name
data_x = data_x.fillna(9999)
veces = data_x['scheme_name'].value_counts()
valores = data_x['scheme_name'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] < 50:
        valores[i] = np.nan
data_x['scheme_name'] = valores.copy()

data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['scheme_name'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 50:
        valores[i] = np.nan
data_x_tst['scheme_name'] = valores.copy()

#funder
data_x = data_x.fillna(9999)
veces = data_x['funder'].value_counts()
valores = data_x['funder'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] < 250:
        valores[i] = np.nan
data_x['funder'] = valores.copy()

data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['funder'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 250:
        valores[i] = np.nan
data_x_tst['funder'] = valores.copy()

#installer
data_x = data_x.fillna(9999)
veces = data_x['installer'].value_counts()
valores = data_x['installer'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] < 200:
        valores[i] = np.nan
data_x['installer'] = valores.copy()


data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['installer'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 300:
        valores[i] = np.nan
data_x_tst['installer'] = valores.copy()

#wpt_name
data_x = data_x.fillna(9999)
veces = data_x['wpt_name'].value_counts()
valores = data_x['wpt_name'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] < 30:
        valores[i] = np.nan
data_x['wpt_name'] = valores.copy()

data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['wpt_name'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 30:
        valores[i] = np.nan
data_x_tst['wpt_name'] = valores.copy()

#subvillage
data_x = data_x.fillna(9999)
veces = data_x['subvillage'].value_counts()
valores = data_x['subvillage'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] < 35:
        valores[i] = np.nan
data_x['subvillage'] = valores.copy()

data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['subvillage'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 35:
        valores[i] = np.nan
data_x_tst['subvillage'] = valores.copy()

#ward
data_x = data_x.fillna(9999)
veces = data_x['ward'].value_counts()
valores = data_x['ward'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces[elem] <70 :
        valores[i] = np.nan
data_x['ward'] = valores.copy()

data_x_tst = data_x_tst.fillna(9999)
valores = data_x_tst['ward'].tolist()
for i,elem in enumerate(valores):
    if elem == 9999:
        valores[i] = np.nan
    elif veces.index.tolist().count(elem) == 0:
        valores[i] = valores[i]
    elif veces[elem] < 70:
        valores[i] = np.nan
data_x_tst['ward'] = valores.copy()

#'''

transformaciones = True

categoricas = [col for col in data_x.columns if data_x[col].dtype=='object']

if transformaciones:

    '''
    Matriz de correlacion
    '''
    '''
    print("------ Eliminacion de variables con matriz de correlacion...")

    EMPEORA
    #Matriz de correlacion
    data_aux = data_x.copy()

    columns = [col for col in data_aux.columns if data_aux[col].dtype=='object']
    columns.append('aux')
    d = create_labelEncoder(data_aux,columns)

    data_aux = categoricas_ordinales(data_aux,d)

    corr_matrix = data_aux.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    umbral = 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > umbral) or any(upper[column] < -1.0 * umbral)]
    data_x = data_x.drop(to_drop, axis=1)
    data_x_tst = data_x_tst.drop(to_drop, axis=1)

    #'''

    '''
    Categoricas a ordinales
    '''
    print("------ Transformación de categoricas...")

    '''
    #data_x_nan,data_x_tst_nan = categoricas_dummy(data_x,data_x_tst,delete=False)

    data_x = data_x.fillna(9999)
    data_x_tst = data_x_tst.fillna(9999)

    data_x,data_x_tst = categoricas_dummy(data_x,data_x_tst)
    #'''


    categoricas.append('aux')
    d = create_labelEncoder(data_x,categoricas)
    categoricas.pop(-1)

    #data_x_nan = data_x.copy()
    #data_x_tst_nan = data_x_tst.copy()
    data_x = data_x.fillna(9999)
    data_x_tst = data_x_tst.fillna(9999)

    data_x = categoricas_ordinales(data_x,d)
    data_x_tst = categoricas_ordinales(data_x_tst,d)

    #data_x_nan = categoricas_ordinales(data_x_nan,d)
    #data_x_tst_nan = categoricas_ordinales(data_x_tst_nan, d)
    #'''
    '''
    Eliminacion de variables que no aportan al modelo, estudiadas anteriormente
    '''

    '''
    print("------ Eliminacion de variables para el modelo...")
    data_x = data_x.drop(['dam45', 'wind-powered15', 'cattle trough42', 'unknown41', 'fluoride29', 'rope pump13', 'unknown19', 'colored28', 'other17', 'other22', 'milky31', 'parastatal18', 'unknown38', 'Ruvuma / Southern Coast7', 'improved spring47', 'pay when scheme fails26', 'Rufiji6', 'Wami / Ruvu8', 'pay annually23', 'Pangani5', 'Lake Rukwa2', 'motorpump11', 'Lake Victoria4', 'salty32', 'Lake Tanganyika3', 'commercial16', 'Lake Nyasa1', 'Internal0', 'user-group20', 'submersible14'],axis=1)
    data_x_tst = data_x_tst.drop(['dam45', 'wind-powered15', 'cattle trough42', 'unknown41', 'fluoride29', 'rope pump13', 'unknown19', 'colored28', 'other17', 'other22', 'milky31', 'parastatal18', 'unknown38', 'Ruvuma / Southern Coast7', 'improved spring47', 'pay when scheme fails26', 'Rufiji6', 'Wami / Ruvu8', 'pay annually23', 'Pangani5', 'Lake Rukwa2', 'motorpump11', 'Lake Victoria4', 'salty32', 'Lake Tanganyika3', 'commercial16', 'Lake Nyasa1', 'Internal0', 'user-group20', 'submersible14'],axis=1)

    #data_x_nan = data_x_nan.drop(['annually32', 'monthly33', 'never pay34', 'on failure35', 'other36', 'per bucket37', 'fluoride abandoned41', 'unknown38', 'dam64', 'good49', 'milky50', 'unknown52', 'colored47', 'wind-powered19', 'fluoride48', 'other21', 'unknown23', 'Ruvuma / Southern Coast7', 'fluoride40', 'unknown60', 'Lake Victoria4', 'coloured39', 'Pangani5', 'parastatal22', 'salty abandoned44', 'rope pump17', 'Lake Nyasa1', 'Lake Tanganyika3', 'Rufiji6', 'salty51', 'other26', 'cattle trough61', 'milky42', 'Lake Rukwa2', 'user-group24', 'Wami / Ruvu8', 'commercial20', 'improved spring66', 'unknown57', 'Internal0','motorpump15', 'pay annually27', 'pay when scheme fails30', 'salty43', 'unknown46', 'surface59', 'gravity13', 'groundwater58', 'submersible18', 'hand pump65', 'False9', 'True12', 'pay monthly28', 'handpump14', 'True10', 'seasonal56', 'False11', 'other16', 'soft45', 'pay per bucket29', 'other67', 'unknown31', 'communal standpipe multiple63', 'dry53', 'communal standpipe62'],axis=1)
    #data_x_tst_nan = data_x_tst_nan.drop(['annually32', 'monthly33', 'never pay34', 'on failure35', 'other36', 'per bucket37', 'fluoride abandoned41', 'unknown38', 'dam64', 'good49', 'milky50', 'unknown52', 'colored47', 'wind-powered19', 'fluoride48', 'other21', 'unknown23', 'Ruvuma / Southern Coast7', 'fluoride40', 'unknown60', 'Lake Victoria4', 'coloured39', 'Pangani5', 'parastatal22', 'salty abandoned44', 'rope pump17', 'Lake Nyasa1', 'Lake Tanganyika3', 'Rufiji6', 'salty51', 'other26', 'cattle trough61', 'milky42', 'Lake Rukwa2', 'user-group24', 'Wami / Ruvu8', 'commercial20', 'improved spring66', 'unknown57', 'Internal0','motorpump15', 'pay annually27', 'pay when scheme fails30', 'salty43', 'unknown46', 'surface59', 'gravity13', 'groundwater58', 'submersible18', 'hand pump65', 'False9', 'True12', 'pay monthly28', 'handpump14', 'True10', 'seasonal56', 'False11', 'other16', 'soft45', 'pay per bucket29', 'other67', 'unknown31', 'communal standpipe multiple63', 'dry53', 'communal standpipe62'],axis=1)
    #'''

    data_x.to_csv('data_x_preproces2.csv', index=False)
    data_x_tst.to_csv('data_x_tst_preproces2.csv', index=False)

else:

    data_x = pd.read_csv('data_x_preproces2.csv')
    data_x_tst = pd.read_csv('data_x_tst_preproces2.csv')
    #'''

#categoricas = [col for col in data_x.columns if data_x[col].dtype=='object']
'''
Procesamos los datos para el ajuste
'''

data_x_xgb = data_x.copy()
data_x_tst_xgb = data_x_tst.copy()

cat = data_x[categoricas]
cat = cat.apply(lambda x: pd.Series(x,dtype='category'))
num = data_x.drop(categoricas,axis=1)
data_x = cat.join(num)

cat = data_x_tst[categoricas]
cat = cat.apply(lambda x: pd.Series(x,dtype='category'))
num = data_x_tst.drop(categoricas,axis=1)
data_x_tst = cat.join(num)

y = np.ravel(data_y.values)

skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=123456)

'''
MODELOS
'''
'''
print("------ XGBoosting...")
clf = xgb.XGBClassifier(n_estimators = 600,objective='multi:softmax',n_jobs=8,max_depth=6)

clfb, y_test_clf = validacion_cruzada(clf,data_x_xgb,y,skf)

clf = clf.fit(data_x_xgb,y)
y_pred_tra = clf.predict(data_x_xgb)
y_pred_tra_clf_prob = clf.predict_proba(data_x_xgb)
print("Score lgb: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst_clf = clf.predict_proba(data_x_tst_xgb)


features = data_x_xgb.columns.values
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#'''

print("------ RandomForest...")
rf = RandomForestClassifier(n_estimators=1000,n_jobs=-1,max_depth=20)

rfb, y_test_svc = validacion_cruzada(rf,data_x,y,skf)


do_grid = False

if do_grid:

    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV

    params_rf = {'n_estimators': [1000, 2000],
                  'max_depth': [8,12],
                  'max_features': ['sqrt','log2'],
                 'n_jobs':[-1],
                 'class_weight':['balanced_subsample',None],
                 'criterion':['gini']}


    print("------ Grid Search...")
    grid = GridSearchCV(rf, params_rf, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(accuracy_score))
    grid.fit(data_x,y)
    print("Mejores parámetros:")
    print(grid.best_params_)
    print("\n------ XGBoosted con los mejores parámetros de GridSearch...")
    gs, y_test_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)

rf = rf.fit(data_x,y)
y_pred_tra = rf.predict(data_x)
y_pred_tra_rf_prob = rf.predict_proba(data_x)
print("Score clf: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst_rf = rf.predict(data_x_tst)
y_pred_tst_rf_prob = rf.predict_proba(data_x_tst)


features = data_x.columns.values
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#'''

'''
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(n_estimators=1500,n_jobs=-1,objective='multiclass')

#lgbmb, y_test_lgb = validacion_cruzada(lgbm,data_x,y,skf)



lgbm = lgbm.fit(data_x,y,feature_name=data_x.columns.tolist())
y_pred_tra = lgbm.predict(data_x)
y_pred_tra_lgb_prob = lgbm.predict_proba(data_x)
print("Score lgb: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst_lgb = lgbm.predict_proba(data_x_tst)

lgb.plot_importance(lgbm)

plt.show()
#'''

'''
print("------ Votos entre RandomForest y LightGBM...")
rf , lgbm , y_test_votos = validacion_cruzada_doble(rf,lgbm,data_x,y,skf)

y_pred_tra = prediccion_conjunta_doble(y_pred_tra_lgb_prob,y_pred_tra_rf_prob)
print("Score multi: {:.4f}".format(accuracy_score(y,y_pred_tra)))

y_pred_tst = prediccion_conjunta_doble(y_pred_tst_lgb,y_pred_tst_rf_prob)
#'''

'''
print("------ Votos entre RandomForest, XGBoosting y LightGBM...")
rf , clf ,lgbm, y_test_votos = validacion_cruzada_triple(rf,clf,lgbm,data_x_xgb,y,skf)

y_pred_tra = prediccion_conjunta_triple(y_pred_tra_clf_prob,y_pred_tra_rf_prob,y_pred_tra_clf_prob)
print("Score multi: {:.4f}".format(accuracy_score(y,y_pred_tra)))

y_pred_tst = prediccion_conjunta_triple(y_pred_tst_clf,y_pred_tst_rf_prob,y_pred_tra_clf_prob)
#'''

'''
#Guardo el resultado en un fichero
df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst_rf
df_submission.to_csv("submissionfinal.csv", index=False)
#'''
print("FINALIZADO\n")