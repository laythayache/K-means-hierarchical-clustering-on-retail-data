import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

def find_optimal_k(X, max_k=10):
    scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.show()
    
    optimal_k = k_range[np.argmax(scores)]
    return optimal_k

def evaluate_clustering(X, labels):
    score = silhouette_score(X, labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"Silhouette Score: {score:.3f}")
    print("Cluster sizes:")
    for i, count in zip(unique, counts):
        print(f"  Cluster {i}: {count}")
    
    return score

def analyze_clusters(features_df, labels):
    df = features_df.copy()
    df['Cluster'] = labels
    
    cluster_summary = df.groupby('Cluster').mean()
    print("\nCluster Profiles:")
    print(cluster_summary)
    
    return cluster_summary

def plot_clusters(features_df, labels, x_col, y_col):
    df = features_df.copy()
    df['Cluster'] = labels
    
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for cluster in np.unique(labels):
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data[x_col], cluster_data[y_col], 
                   c=colors[cluster % len(colors)], label=f'Cluster {cluster}')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title(f'{x_col} vs {y_col}')
    plt.show()

def compare_methods(features_df, kmeans_labels, hierarchical_labels):
    print("Comparison of Clustering Methods:")
    
    kmeans_score = silhouette_score(features_df, kmeans_labels)
    hierarchical_score = silhouette_score(features_df, hierarchical_labels)
    
    print(f"K-means Silhouette Score: {kmeans_score:.3f}")
    print(f"Hierarchical Silhouette Score: {hierarchical_score:.3f}")
    
    crosstab = pd.crosstab(kmeans_labels, hierarchical_labels)
    print("\nCluster Assignment Comparison:")
    print(crosstab)
    
    if kmeans_score > hierarchical_score:
        print("K-means performs better")
        return 'kmeans'
    else:
        print("Hierarchical performs better")
        return 'hierarchical'
