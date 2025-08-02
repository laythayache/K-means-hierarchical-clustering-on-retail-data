import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_data, clean_data, create_features, scale_features
from evaluation import find_optimal_k, evaluate_clustering, analyze_clusters, plot_clusters, compare_methods

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def run_hierarchical(X, n_clusters):
    linkage_matrix = linkage(X, method='ward')
    
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()
    
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    return labels

def generate_insights(features_df, labels):
    df = features_df.copy()
    df['Cluster'] = labels
    
    insights = {}
    overall_recency = df['Recency'].mean()
    overall_frequency = df['Frequency'].mean()
    overall_spending = df['TotalSpent'].mean()
    
    for cluster in np.unique(labels):
        cluster_data = df[df['Cluster'] == cluster]
        
        avg_recency = cluster_data['Recency'].mean()
        avg_frequency = cluster_data['Frequency'].mean()
        avg_spending = cluster_data['TotalSpent'].mean()
        
        segment_name = "Regular"
        if avg_spending > overall_spending * 1.5 and avg_frequency > overall_frequency:
            segment_name = "VIP"
        elif avg_recency > overall_recency * 2:
            segment_name = "At-Risk"
        elif avg_frequency > overall_frequency:
            segment_name = "Loyal"
        
        insights[cluster] = {
            'name': segment_name,
            'size': len(cluster_data),
            'recency': avg_recency,
            'frequency': avg_frequency,
            'spending': avg_spending
        }
    
    return insights

def main():
    data_path = "data/raw/OnlineRetail.csv"
    
    print("Loading and preprocessing data...")
    raw_data = load_data(data_path)
    clean_df = clean_data(raw_data)
    features = create_features(clean_df)
    scaled_features, scaler = scale_features(features)
    
    print("Finding optimal number of clusters...")
    optimal_k = find_optimal_k(scaled_features, max_k=8)
    print(f"Optimal k: {optimal_k}")
    
    print("Running K-means clustering...")
    kmeans_labels, kmeans_model = run_kmeans(scaled_features, optimal_k)
    kmeans_score = evaluate_clustering(scaled_features, kmeans_labels)
    
    print("Running Hierarchical clustering...")
    hierarchical_labels = run_hierarchical(scaled_features, optimal_k)
    hierarchical_score = evaluate_clustering(scaled_features, hierarchical_labels)
    
    print("Comparing methods...")
    best_method = compare_methods(features, kmeans_labels, hierarchical_labels)
    
    if best_method == 'kmeans':
        best_labels = kmeans_labels
    else:
        best_labels = hierarchical_labels
    
    print("Generating business insights...")
    insights = generate_insights(features, best_labels)
    
    print("\nCustomer Segments:")
    for cluster, info in insights.items():
        print(f"Cluster {cluster} ({info['name']}): {info['size']} customers")
        print(f"  Avg Recency: {info['recency']:.1f} days")
        print(f"  Avg Frequency: {info['frequency']:.1f}")
        print(f"  Avg Spending: ${info['spending']:.0f}")
    
    plot_clusters(features, best_labels, 'Recency', 'TotalSpent')
    plot_clusters(features, best_labels, 'Frequency', 'AvgOrderValue')
    
    results_df = features.copy()
    results_df['Cluster'] = best_labels
    results_df.to_csv("results/clustering_results.csv")
    
    print("Analysis complete! Results saved to results/clustering_results.csv")
    return results_df

if __name__ == "__main__":
    results = main()
