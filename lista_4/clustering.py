import os
import sys
import numpy as np 
from time import sleep
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

PARENT_PATH = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(PARENT_PATH, "results")

TITLE_FONTDICT = {'fontsize': 18,
                  'fontweight' : 'bold',
                  'verticalalignment': 'bottom',
                  'horizontalalignment': 'center'}
SUBTITLE_FONTDICT = {'fontsize': 12,
                     'fontweight' : 'normal',
                     'verticalalignment': 'bottom',
                     'horizontalalignment': 'center'}


def save_plot(x_values, y_values, x_label, y_label, filename, title, subtitle=None):
    plt.style.use("fivethirtyeight")
    plt.subplots(figsize=(10,8))
    plt.plot(x_values, y_values)
    plt.xticks(x_values)
    plt.suptitle(title, fontdict=TITLE_FONTDICT,  y=1.08)
    if subtitle is not None:
        plt.title(subtitle,fontdict=SUBTITLE_FONTDICT, y=1.03)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    sleep(3)
    plt.close()

def get_kstar(features, class_label=None, dataset_name=None, n_fold=None):
    k_values = [2,3,4,5,6]
    sse_values = []
    silhouette_coefficients = []
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    print("results_dir =", results_dir)

    for k in k_values:
        kmeans = KMeans(n_clusters=k
                        ,init='k-means++'
                        ,n_init=10
                        ,max_iter=300
                        ,random_state=42)
        
        kmeans.fit(features)

        # Getting SSE value
        sse = kmeans.inertia_
        sse_values.append(sse)

        # Getting Silhouette Coefficient value
        preds = kmeans.labels_
        score = silhouette_score(features, preds)
        silhouette_coefficients.append(score)
        
    # The Elbow Method
    kl = KneeLocator(k_values, sse_values, curve="convex", direction="decreasing")
    k_elbow = kl.elbow
    filename = os.path.join(results_dir, f"{dataset_name}_elbow_{class_label}_fold{n_fold}.png")
    save_plot(x_values=k_values,
              y_values=sse_values,
              x_label="Número de Clusters (k)",
              y_label="Soma dos Erros ao Quadrado (SSE)",
              filename=filename,
              title="SSE por Número de Clusters",
              subtitle="Classe %d" % class_label)
    
    # The Silhouette Method
    if k_elbow is not None:
        k_star = k_elbow
    else:
        k_star = 3

    if silhouette_coefficients[k_star - 2] < silhouette_coefficients[k_star - 1]:
        k_star = k_star + 1
    filename = os.path.join(results_dir, f"{dataset_name}_silhouette_{class_label}_fold{n_fold}.png")
    save_plot(x_values=k_values,
              y_values=silhouette_coefficients,
              x_label="Número de Clusters (k)",
              y_label="Coeficiente de Silhouette",
              filename=filename,
              title="Coeficiente de Silhouette por Número de Clusters",
              subtitle="Classe %d" % class_label)

    return k_star

def generate_kstar_clusters(features, class_label, dataset_name=None, n_fold=None):
    k_star = get_kstar(features, class_label, dataset_name, n_fold)
    kmeans = KMeans(n_clusters=k_star
                    ,init='k-means++'
                    ,n_init=10
                    ,max_iter=300
                    ,random_state=42)
        
    kmeans.fit(features)
    y_new = kmeans.labels_

    return k_star, y_new
