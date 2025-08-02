# K-means & Hierarchical Clustering on Retail Data

Customer segmentation analysis using K-means and Hierarchical clustering algorithms on retail transaction data.

## Overview

This project performs customer segmentation analysis on retail transaction data to identify distinct customer groups. The analysis compares K-means and Hierarchical clustering methods to find the optimal segmentation strategy.

## Dataset

Uses the Online Retail Dataset containing transactions from a UK-based online retail company.

Dataset Features:
- InvoiceNo: Transaction identifier
- StockCode: Product code
- Description: Product description
- Quantity: Product quantity
- InvoiceDate: Transaction date
- UnitPrice: Product price
- CustomerID: Customer identifier
- Country: Customer country

## Project Structure

```
├── data/
│   ├── raw/OnlineRetail.csv         # Raw dataset
│   └── processed/                   # Processed datasets
├── src/
│   ├── preprocessing.py             # Data preprocessing functions
│   ├── evaluation.py                # Clustering evaluation functions
│   ├── implementation.py            # Main clustering implementation
│   └── __init__.py                  # Package initialization
├── results/                         # Generated results
├── main.py                          # Main execution script
├── quick_start.py                   # Simple example script
└── requirements.txt                 # Dependencies
```

## Usage

### Complete Analysis
```bash
python main.py
```

### Quick Demo
```bash
python quick_start.py
```

## Methodology

### Data Preprocessing
- Remove transactions with missing CustomerID
- Filter out negative quantities and prices
- Handle extreme outliers using percentile capping
- Create customer-level features from transaction data

### Feature Engineering
Creates customer features:
- Recency: Days since last purchase
- Frequency: Number of transactions
- Total Spent: Total monetary value
- Average Order Value: Mean transaction amount
- Total Quantity: Total items purchased
- Number of Sessions: Unique shopping sessions

### Clustering Methods
- K-means clustering with optimal k determination
- Hierarchical clustering using Ward linkage
- Performance comparison using silhouette analysis

### Evaluation Metrics
- Silhouette Score: Measures cluster cohesion and separation
- Cluster size distribution
- Business relevance assessment

## Results

The analysis typically identifies customer segments such as:

1. Regular Customers: Moderate engagement and spending
2. VIP Customers: High spending and frequent purchases
3. At-Risk Customers: Declining activity patterns

Results are saved to CSV files in the results/ directory.

## Business Applications

- Customer segmentation for targeted marketing
- Customer lifetime value analysis
- Retention strategy development
- Product recommendation systems

## Files Description

- **main.py**: Complete analysis pipeline with detailed output
- **quick_start.py**: Simplified analysis for quick results
- **src/preprocessing.py**: Data cleaning and feature engineering
- **src/evaluation.py**: Clustering evaluation and metrics
- **src/implementation.py**: Core clustering algorithms and insights

## Output Files

- results/clustering_results.csv: Detailed customer cluster assignments
- results/quick_results.csv: Basic clustering results
