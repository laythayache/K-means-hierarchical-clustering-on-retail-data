import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.implementation import main as run_clustering

if __name__ == "__main__":
    print("Running K-means and Hierarchical Clustering Analysis...")
    
    if not os.path.exists("data/raw/OnlineRetail.csv"):
        print("Error: OnlineRetail.csv not found in data/raw/")
        print("Please make sure the data file is in the correct location.")
        sys.exit(1)
    
    try:
        results = run_clustering()
        if results is not None:
            print("Analysis completed successfully!")
        else:
            print("Analysis failed.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the project root directory.")
