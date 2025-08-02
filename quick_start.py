import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import load_data, clean_data, create_features, scale_features
from src.evaluation import find_optimal_k, evaluate_clustering, analyze_clusters
from src.implementation import run_kmeans, run_hierarchical, generate_insights

data_path = "data/raw/OnlineRetail.csv"

raw_data = load_data(data_path)
clean_df = clean_data(raw_data)
features = create_features(clean_df)
scaled_features, scaler = scale_features(features)

optimal_k = find_optimal_k(scaled_features, max_k=6)
print(f"Best k: {optimal_k}")

kmeans_labels, kmeans_model = run_kmeans(scaled_features, optimal_k)
hierarchical_labels = run_hierarchical(scaled_features, optimal_k)

kmeans_score = evaluate_clustering(scaled_features, kmeans_labels)
hierarchical_score = evaluate_clustering(scaled_features, hierarchical_labels)

if kmeans_score > hierarchical_score:
    best_labels = kmeans_labels
    print("K-means is better")
else:
    best_labels = hierarchical_labels
    print("Hierarchical is better")

insights = generate_insights(features, best_labels)
print("\nCustomer Segments:")
for cluster, info in insights.items():
    print(f"{info['name']}: {info['size']} customers")

results_df = features.copy()
results_df['Cluster'] = best_labels
results_df.to_csv("results/quick_results.csv")
print("Done! Check results/quick_results.csv")
