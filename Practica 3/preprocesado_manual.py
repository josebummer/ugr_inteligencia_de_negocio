#Leemos los datos
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

data_x = pd.read_csv('water_pump_tra.csv')
data_x_tst = pd.read_csv('water_pump_tst.csv')


def knn_imputation(data_x,data_x_tst,columns,col,K=5,clasificacion=False):
    # Train
    if clasificacion:
        nulos = data_x.isna()
        index_no_null = nulos.index[nulos[col]==False]
        data_x_n = data_x.loc[index_no_null]
    else:
        data_x_n = data_x.loc[data_x[col] > 0]
    columns.append(col)
    data_x_n = data_x_n[columns]

    aux = data_x_n.isna()
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_n = data_x_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)

    aux = data_x_n == 0
    for colum in aux.columns:
        index = aux.index[aux[colum] == True]
        if len(index) > 0:
            data_x_n = data_x_n.drop(index, axis=0)
            aux = aux.drop(index, axis=0)

    data_y_n = data_x_n[col].values.ravel()
    data_x_n = data_x_n.drop(col, axis=1)

    data_x_n = pd.DataFrame(MinMaxScaler().fit_transform(data_x_n), index=data_x_n.index, columns=data_x_n.columns)

    if clasificacion:
        nulos = data_x.isna()
        index_null = nulos.index[nulos[col] == True]
        nulos = data_x.loc[index_null]
    else:
        nulos = data_x.loc[data_x[col] == 0]
    nulos = nulos[columns]
    nulos = nulos.drop(col, axis=1)

    if clasificacion:
        neight = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_n, data_y_n.astype(int))
    else:
        neight = KNeighborsRegressor(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_n, data_y_n)

    #nulos = pd.DataFrame(MinMaxScaler().fit_transform(nulos), index=nulos.index, columns=nulos.columns)
    for colum in nulos.columns:
        if (nulos[colum].max() - nulos[colum].min()) == 0:
            nulos[colum] = 0.0
        else:
            nulos[colum] = (nulos[colum] - nulos[colum].min()) / (nulos[colum].max() - nulos[colum].min())

    labels = neight.predict(nulos)

    data_x.iloc[nulos.index, data_x.columns.get_loc(col)] = labels

    # Test
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

    data_x_tst_n = pd.DataFrame(MinMaxScaler().fit_transform(data_x_tst_n), index=data_x_tst_n.index,
                                columns=data_x_tst_n.columns)

    if clasificacion:
        nulos = data_x_tst.isna()
        index_null = nulos.index[nulos[col] == True]
        nulos = data_x_tst.loc[index_null]
    else:
        nulos = data_x_tst.loc[data_x_tst[col] == 0]
    nulos = nulos[columns]
    nulos = nulos.drop(col, axis=1)

    if clasificacion:
        neight = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_tst_n, data_y_n.astype(int))
    else:
        neight = KNeighborsRegressor(n_neighbors=K, n_jobs=-1)
        neight.fit(data_x_tst_n, data_y_n)

    #nulos = pd.DataFrame(MinMaxScaler().fit_transform(nulos), index=nulos.index, columns=nulos.columns)
    for colum in nulos.columns:
        if (nulos[colum].max() - nulos[colum].min()) == 0:
            nulos[colum] = 0.0
        else:
            nulos[colum] = (nulos[colum] - nulos[colum].min()) / (nulos[colum].max() - nulos[colum].min())

    labels = neight.predict(nulos)

    data_x_tst.iloc[nulos.index, data_x_tst.columns.get_loc(col)] = labels

'''
ID
'''
#Eliminamos id porque no es necesaria
data_x = data_x.drop('id',axis=1)
data_x_tst = data_x_tst.drop('id',axis=1)

'''
scheme_name
'''
#Eliminamos esta variable porque casi la mitad de los valores son nulos y porque tenemos scheme_managent
data_x = data_x.drop('scheme_name',axis=1)
data_x_tst = data_x_tst.drop('scheme_name',axis=1)

'''
amount_tsh
'''
#Eliminamos amount_tsh porque tiene un 50% de valores nulos
data_x = data_x.drop('amount_tsh',axis=1)
data_x_tst = data_x_tst.drop('amount_tsh',axis=1)

'''
num_private
'''
#Eliminamos esta variable porque no aporta nada
data_x = data_x.drop('num_private',axis=1)
data_x_tst = data_x_tst.drop('num_private',axis=1)

'''
recorded_by
'''
#Se elimina porque para todos los valores vale lo mismo
data_x = data_x.drop('recorded_by',axis=1)
data_x_tst = data_x_tst.drop('recorded_by',axis=1)


'''
date_recorded
'''
#Pasamos date_recorded a una columna interpretable
#Train
date = pd.to_datetime(data_x['date_recorded'])
anio = date.apply(lambda x: x.year)
mes = date.apply(lambda x: x.month)
ordinal = date.apply(lambda x: x.toordinal())

data_x['year_recorded'] = anio
data_x['mes_recorded'] = mes
data_x['date_recorded'] = ordinal

#Test
date = pd.to_datetime(data_x_tst['date_recorded'])
anio = date.apply(lambda x: x.year)
mes = date.apply(lambda x: x.month)
ordinal = date.apply(lambda x: x.toordinal())

data_x_tst['year_recorded'] = anio
data_x_tst['mes_recorded'] = mes
data_x_tst['date_recorded'] = ordinal


'''
longitude y latitude
'''
'''
#Train
not_null = data_x.loc[data_x['longitude'] > 0]
vacios_ward = data_x.loc[data_x['longitude']==0]['ward'].unique()
vacios_lga = data_x.loc[data_x['longitude']==0]['lga'].unique()
vacios_region = data_x.loc[data_x['longitude']==0]['region'].unique()

for vw,vl,vr in zip(vacios_ward,vacios_lga,vacios_region):
    igualesw = not_null.loc[not_null['ward']==vw].values
    igualesl = not_null.loc[not_null['lga'] == vl].values
    igualesr = not_null.loc[not_null['region'] == vr].values
    if len(igualesw) > 0:
        media1 = not_null.loc[not_null['ward']==vw]['longitude'].mean()
        media2 = not_null.loc[not_null['ward'] == vw]['latitude'].mean()
        index = data_x.loc[data_x['longitude']==0]
        index = index.loc[index['ward']==vw].index
        data_x.iloc[index, data_x.columns.get_loc('longitude')] = media1
        data_x.iloc[index, data_x.columns.get_loc('latitude')] = media2
    elif len(igualesl) > 0:
        media1 = not_null.loc[not_null['lga'] == vl]['longitude'].mean()
        media2 = not_null.loc[not_null['lga'] == vl]['latitude'].mean()
        index = data_x.loc[data_x['longitude'] == 0]
        index = index.loc[index['lga'] == vl].index
        data_x.iloc[index, data_x.columns.get_loc('longitude')] = media1
        data_x.iloc[index, data_x.columns.get_loc('latitude')] = media2
    else:
        media1 = not_null.loc[not_null['region'] == vr]['longitude'].mean()
        media2 = not_null.loc[not_null['region'] == vr]['latitude'].mean()
        index = data_x.loc[data_x['longitude'] == 0]
        index = index.loc[index['region'] == vr].index
        data_x.iloc[index, data_x.columns.get_loc('longitude')] = media1
        data_x.iloc[index, data_x.columns.get_loc('latitude')] = media2

#Test
not_null = data_x_tst.loc[data_x_tst['longitude'] > 0]
vacios_ward = data_x_tst.loc[data_x_tst['longitude']==0]['ward'].unique()
vacios_lga = data_x_tst.loc[data_x_tst['longitude']==0]['lga'].unique()
vacios_region = data_x_tst.loc[data_x_tst['longitude']==0]['region'].unique()

for vw,vl,vr in zip(vacios_ward,vacios_lga,vacios_region):
    igualesw = not_null.loc[not_null['ward']==vw].values
    igualesl = not_null.loc[not_null['lga'] == vl].values
    igualesr = not_null.loc[not_null['region'] == vr].values
    if len(igualesw) > 0:
        media1 = not_null.loc[not_null['ward']==vw]['longitude'].mean()
        media2 = not_null.loc[not_null['ward'] == vw]['latitude'].mean()
        index = data_x_tst.loc[data_x_tst['longitude']==0]
        index = index.loc[index['ward']==vw].index
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('longitude')] = media1
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('latitude')] = media2
    elif len(igualesl) > 0:
        media1 = not_null.loc[not_null['lga'] == vl]['longitude'].mean()
        media2 = not_null.loc[not_null['lga'] == vl]['latitude'].mean()
        index = data_x_tst.loc[data_x_tst['longitude'] == 0]
        index = index.loc[index['lga'] == vl].index
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('longitude')] = media1
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('latitude')] = media2
    else:
        media1 = not_null.loc[not_null['region'] == vr]['longitude'].mean()
        media2 = not_null.loc[not_null['region'] == vr]['latitude'].mean()
        index = data_x_tst.loc[data_x_tst['longitude'] == 0]
        index = index.loc[index['region'] == vr].index
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('longitude')] = media1
        data_x_tst.iloc[index, data_x_tst.columns.get_loc('latitude')] = media2
#'''

'''
installer
public_meeting
scheme_management
'''

from sklearn.preprocessing import LabelEncoder
import numpy as np

#Train
strings = [col for col in data_x.columns if data_x[col].dtype=='object']
df_strings = pd.DataFrame(data_x[strings],index=data_x.index,columns=strings)
numbers = np.setdiff1d(data_x.columns.values,np.array(strings))
df_numbers = pd.DataFrame(data_x[numbers], index=data_x.index, columns=numbers)

mask = df_strings.isna() #mÃ¡scara para luego recuperar los NaN
data_x_tmp = df_strings.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categÃ³ricas en numÃ©ricas
df_strings = data_x_tmp.where(~mask, df_strings) #se recuperan los NaN
data_x = pd.DataFrame.join(df_strings,df_numbers)

#Test
strings = [col for col in data_x_tst.columns if data_x_tst[col].dtype=='object']
df_strings = pd.DataFrame(data_x_tst[strings],index=data_x_tst.index,columns=strings)
numbers = np.setdiff1d(data_x_tst.columns.values,np.array(strings))
df_numbers = pd.DataFrame(data_x_tst[numbers], index=data_x_tst.index, columns=numbers)

mask = df_strings.isna() #mÃ¡scara para luego recuperar los NaN
data_x_tmp = df_strings.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categÃ³ricas en numÃ©ricas
df_strings = data_x_tmp.where(~mask, df_strings) #se recuperan los NaN

data_x_tst = pd.DataFrame.join(df_strings,df_numbers)
#'''

'''
gps_height
'''

knn_imputation(data_x,data_x_tst,['longitude','latitude','extraction_type_class','district_code'],'gps_height')

'''
subvillage
'''
knn_imputation(data_x,data_x_tst,['basin','latitude','longitude','gps_height','region_code','district_code','population'],'subvillage',clasificacion=True)

'''
scheme_management
'''
knn_imputation(data_x,data_x_tst,['date_recorded','mes_recorded','year_recorded','latitude','longitude','gps_height','district_code','subvillage','basin','region_code','ward'],'scheme_management',clasificacion=True)

'''
funder
'''
knn_imputation(data_x,data_x_tst,['scheme_management','latitude','longitude','gps_height','district_code','subvillage','construction_year'],'funder',clasificacion=True)

'''
scheme_name
'''
knn_imputation(data_x,data_x_tst,['funder','scheme_management','latitude','longitude','gps_height','district_code','subvillage','construction_year'],'scheme_name',clasificacion=True)


'''
construction_year
'''

knn_imputation(data_x,data_x_tst,['subvillage','latitude','longitude','date_recorded','mes_recorded','year_recorded'],'construction_year')


'''
population
'''

knn_imputation(data_x,data_x_tst,['date_recorded','mes_recorded','year_recorded','water_quality','ward','region','construction_year','quantity','subvillage'],'population')

'''
amount_tsh
'''
knn_imputation(data_x,data_x_tst,['population','ward','construction_year','subvillage','date_recorded','mes_recorded','year_recorded','longitude','latitude','gps_height','quantity','water_quality','waterpoint_type','source_class'],'amount_tsh')


'''
permit
'''
knn_imputation(data_x,data_x_tst,['date_recorded','year_recorded','mes_recorded','construction_year','population','management','payment','water_quality','quantity','source_type','source_class','waterpoint_type','amount_tsh','subvillage'],'permit',clasificacion=True)


# Eliminamos las variables que tengan un nivel alto de correlacion
corr_matrix = data_x.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
umbral = 0.9
to_drop = [column for column in upper.columns if any(upper[column] > umbral) or any(upper[column] < -1.0 * umbral)]
# to_drop = ['extraction_type_group', 'quantity_group', 'source_type', 'waterpoint_type_group']
data_x = data_x.drop(to_drop, axis=1)
data_x_tst = data_x_tst.drop(to_drop, axis=1)
#'''

#Guardo el resultado en un fichero
data_x.to_csv('data_x_preproces.csv',index=False)
data_x_tst.to_csv('data_x_tst_preproces.csv',index=False)

print("FINALIZADO\n")