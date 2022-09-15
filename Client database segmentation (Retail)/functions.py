import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from math import radians, cos, sin, asin, sqrt

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center',
                                 rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

            
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    
def PCA(data, n_comp = 6, show_corr_circles=False, show_fact_planes=False, fill_nan_method=None, fill_value=None,
        dataframe=False, row_names=None, return_values=False):

    if dataframe:
        features = np.array(data.columns)
        names = np.array(data.index)
    if row_names:
        names = np.array(data[row_names])
    if fill_nan_method:
        data = SimpleImputer(strategy = fill_nan_method).fit_transform(data)
        
    data = preprocessing.StandardScaler().fit_transform(data)
    pca = decomposition.PCA(n_components=n_comp)
    feat_pca = pca.fit_transform(data)
    print("Dataset dimensions before PCA : ", data.shape)
    print("Dataset dimensions after PCA : ", feat_pca.shape)
    display_scree_plot(pca)
    
    if (len(pca.components_) <= 10):
        axis_ranks = []
        numb_comp = len(pca.components_)
        for k in range(int(numb_comp/2)):
            a, b = 2*k, 2*k+1
            axis_ranks.append([a, b])
        if show_corr_circles & dataframe:
            display_circles(pca.components_, numb_comp, pca, axis_ranks, labels = features)
        elif show_corr_circles:
            display_circles(pca.components_, numb_comp, pca, axis_ranks, labels = None)
        if show_fact_planes & dataframe:
            display_factorial_planes(feat_pca, numb_comp, pca, axis_ranks, labels = names)
        elif show_fact_planes:
            display_factorial_planes(feat_pca, numb_comp, pca, axis_ranks, labels = None)
            
    elif show_corr_circles | show_fact_planes:
        print('Too many dimensions to display correlation circles or factorial planes')
        
    if return_values:
        return feat_pca, pca.components_

    
def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
    
    
def anova(df, category, variable, showfliers=True, figsize=(18,7)):
    var_mean=df[variable].mean()
    fig=plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    plt.title("{} by {}".format(variable, category),size=16)
    sns.boxplot(x=category, y=variable, data=df,color="#cbd1db",width=0.5,showfliers=showfliers,showmeans=True)
    plt.hlines(y=var_mean,xmin=-0.5,xmax=len(df[category].unique())-0.5,color="#6d788b",ls="--",label="Global mean")

    plt.xticks(range(0,len(df[category].unique()))
               ,df[category].unique(),rotation=90)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def corr_heatmap(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={'size':10}, 
                     mask=mask, center=0, cmap='coolwarm')
    plt.title('Linear Correlation Heatmap\n', fontsize = 16)
    plt.show()
    
    
def haversine_distance(lat1, lng1, lat2, lng2, degrees=True):
    r = 6371
    if degrees:
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlng = lng2 - lng1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    d = 2 * r * asin(sqrt(a))  
    return d