#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:18:29 2018

@author: jose
"""

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from scipy.cluster import hierarchy
from sklearn import metrics
#from sklearn import preprocessing
from math import floor
import seaborn as sns

# ### Funciones

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

def readCSV(n):
    name = n+".csv"

    data = pd.read_csv(name)
    data = data.replace(np.NaN,0) #los valores en blanco realmente son otra categoría que nombramos como 0
    
    return data

def plotScatterMatrix(df,n):
    name = n+".png"
    #'''
    print("---------- Preparando el scatter matrix...")
    sns.set()
    variables = list(df)
    variables.remove('cluster')
    sns_plot = sns.pairplot(df, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig(name)
    plt.close()
    print("")
    #'''
    
def plotCorrelationMatrix(df,name):
    #'''
    print("---------- Preparando el correlation matrix...")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title(name)
    labels=['NULL']
    labels.extend(df.keys())
    ax1.set_xticklabels(labels,fontsize=8)
    ax1.set_yticklabels(labels,fontsize=8)
    fig.colorbar(cax)
    plt.savefig(name+".png")
    plt.close()
    print("")
    #'''
    
    
def kMeans(data,clusters=8): 
    print('----- Ejecutando k-Means',end='')
    k_means = KMeans(init='k-means++', n_clusters=clusters)
    t = time.time()
    cluster_predict = k_means.fit_predict(data) 
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(data, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
    metric_SC = metrics.silhouette_score(data, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    
    return k_means,clusters

def agglomerativeClustering(data,clusters=2,linkage="ward",affinity="euclidean"):
    print('----- Ejecutando AgglomerativeClustering',end='')
    clustering = AgglomerativeClustering(n_clusters=clusters,linkage=linkage,affinity=affinity)
    t = time.time()
    cluster_predict = clustering.fit_predict(data)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(data, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
    metric_SC = metrics.silhouette_score(data, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    
    return clustering,clusters

def dbsSCAN(data,eps=0.24,clusters=None):
    print('----- Ejecutando DBSCAN',end='')
    db = DBSCAN(eps=eps)
    t = time.time()
    cluster_predict = db.fit_predict(data)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(data, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
    metric_SC = metrics.silhouette_score(data, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    
    return db,clusters

def birch(data,factor=50,clusters=3,threashold=0.5):
    print('----- Ejecutando BRICH',end='')
    bir = Birch(branching_factor=factor, n_clusters=clusters, threshold=threashold,compute_labels=True)
    t = time.time()
    cluster_predict = bir.fit_predict(data)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(data, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
    metric_SC = metrics.silhouette_score(data, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    
    return bir,clusters

def spectralClustering(data,clusters=4):
    print('----- Ejecutando SpectralClustering',end='')
    sp = SpectralClustering(n_clusters=clusters,assign_labels="discretize")
    t = time.time()
    cluster_predict = sp.fit_predict(data)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(data, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
    metric_SC = metrics.silhouette_score(data, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    
    return sp,clusters

def plotHeatMap(k_means,clusters,X,n):
    name = n+".png"
    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
    centers_desnormal = centers.copy()
    
    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
    
    sns_plot = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    fig = sns_plot.get_figure()
    fig.savefig(name)
    plt.close()
    
def createClusters(clusters,data):    
    indexs = []
    for i in clusters.groupby(['cluster']).count().index.tolist():
        indexs.append(clusters.loc[clusters['cluster']==i].index)
        
    clustersl = []
    for i in indexs:
        clustersl.append(data.loc[i])

    return clustersl

def tamClusters(clusters):
    print("Tamaño de cada cluster:")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
       print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
       

def plotDendograma(data,n):
    if(len(data) > 1000):
        d = data.sample(1000)
    else:
        d = data
    name = n+".png"
    linkage_array = hierarchy.ward(d)
    plt.figure(1)
    plt.clf()
    hierarchy.dendrogram(linkage_array,orientation='left')
    
    #Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
    sns_plot = sns.clustermap(d, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
    sns_plot.savefig(name)
    plt.close()
    
def filtrarDatos(X_ag,min_size):
    # Agrupamos las instancias de 'X_cluster' por el cluster asignado 
    X_filtrado = X_ag[X_ag.groupby('cluster').cluster.transform(len) > min_size]
    
    X_res = X_filtrado
    X_filtrado = X_filtrado.drop('cluster', axis=1)
    
    return X_res,X_filtrado

# ### Código
      
censo = readCSV('censo_granada')

# =============================================================================
## CASO 1
# =============================================================================

#Caso de estudiantes que tenemos informacion sobre los estudios de sus padres
subset1 = censo.loc[censo['ESCUR1']>0]
subset2 = censo.loc[censo['ESCUR2']>0]
subset3 = censo.loc[censo['ESCUR3']>0]
subset = pd.concat([subset1,subset2,subset3]).drop_duplicates().reset_index(drop=True)
subset = subset.loc[subset['EDAD']>17]
subset = subset.loc[subset['ESTUPAD']>0]
subset = subset.loc[subset['ESTUMAD']>0]

#seleccionar variables de interés para clustering
usadas = ['ESCUR1','ESTUPAD','ESTUMAD','TENEN','EDAD']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

NCLUSTERS = 7

clustering_algorithms = (
        ('KMeans', kMeans),
        ('AgglomerativeClustering', agglomerativeClustering),
        ('dbsSCAN', dbsSCAN),
        ('birch', birch),
        ('spectralClustering', spectralClustering)
        )

# =============================================================================
# All Algorithm - 1
# =============================================================================

for name,alg in clustering_algorithms:
    mod,clusters = alg(data=X_normal)
    datos = pd.concat([X, clusters], axis=1)
    datos_normal = pd.concat([X_normal, clusters], axis=1)
    
    if  name == 'AgglomerativeClustering':
        datos_normal,filtrado = filtrarDatos(datos_normal,3)
        clusters = datos_normal[['cluster']]
        plotDendograma(filtrado,"dendograma-1")
    
    tamClusters(clusters)
     
    if(NCLUSTERS < 11):
        if name == 'KMeans':
            plotHeatMap(mod,clusters,X,"heat"+name+"-1")
         
        plotScatterMatrix(datos,name+"-1")
         
        clustersi = createClusters(clusters,X_normal)
         
        for i,c in enumerate(clustersi):    
            plotCorrelationMatrix(c,"cluster-"+name+str(i)+"-1")


# =============================================================================
## CASO 2
# =============================================================================
            
#Caso para personas mayores de 29 años que su pareja tiene estudios,
# que vivien en casa propia, que tienen hijos y que es
# empresario/a.
subset = censo.loc[censo['EDAD']>29]
subset = subset.loc[subset['ESTUCON']>0]
subset = subset.loc[subset['TENEN']<4]
subset = subset.loc[subset['TIPONUC']==2]
subset = subset.loc[subset['SITU']==1]

#seleccionar variables de interés para clustering
usadas = ['EDAD','ESREAL','ESTUCON','NMIEM','NOCU']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

clustering_algorithms = (
        ('KMeans', kMeans),
        ('AgglomerativeClustering', agglomerativeClustering),
        ('dbsSCAN', dbsSCAN),
        ('birch', birch),
        ('spectralClustering', spectralClustering)
        )

# =============================================================================
# All Algorithm - 2
# =============================================================================

NCLUSTERS=7

for name,alg in clustering_algorithms:
    mod,clusters = alg(data=X_normal)
    datos = pd.concat([X, clusters], axis=1)
    datos_normal = pd.concat([X_normal, clusters], axis=1)
    
    if  name == 'AgglomerativeClustering':
        datos_normal,filtrado = filtrarDatos(datos_normal,3)
        clusters = datos_normal[['cluster']]
        plotDendograma(filtrado,"dendograma-2")
    
    tamClusters(clusters)
     
    if(NCLUSTERS < 11):
        if name == 'KMeans':
            plotHeatMap(mod,clusters,X,"heat"+name+"-2")
         
        plotScatterMatrix(datos,name+"-2")
         
        clustersi = createClusters(clusters,X_normal)
         
        for i,c in enumerate(clustersi):    
            plotCorrelationMatrix(c,"cluster-"+name+str(i)+"-2")
            
# =============================================================================
## CASO 3
# =============================================================================
            
#Caso para        

subset = censo.loc[censo['TAREA2']==1]
subset = subset.loc[censo['LTRABA']==4]

#seleccionar variables de interés para clustering
usadas = ['EDAD','NVIAJE','TDESP','NMIEM','H6584']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

clustering_algorithms = (
        ('KMeans', kMeans),
        ('AgglomerativeClustering', agglomerativeClustering),
        ('dbsSCAN', dbsSCAN),
        ('birch', birch),
        ('spectralClustering', spectralClustering)
        )

# =============================================================================
# All Algorithm - 3
# =============================================================================

NCLUSTERS=7

for name,alg in clustering_algorithms:
    mod,clusters = alg(data=X_normal)
    datos = pd.concat([X, clusters], axis=1)
    datos_normal = pd.concat([X_normal, clusters], axis=1)
    
    if  name == 'AgglomerativeClustering':
        datos,filtrado = filtrarDatos(datos_normal,3)
        clusters = datos_normal[['cluster']]
        plotDendograma(filtrado,"dendograma-3")
    
    tamClusters(clusters)
     
    if(NCLUSTERS < 11):
        if name == 'KMeans':
            plotHeatMap(mod,clusters,X,"heat"+name+"-3")
         
        plotScatterMatrix(datos,name+"-3")
         
        clustersi = createClusters(clusters,X_normal)
         
        for i,c in enumerate(clustersi):    
            plotCorrelationMatrix(c,"cluster-"+name+str(i)+"-3")
