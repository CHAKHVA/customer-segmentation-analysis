# Customer Segmentation Analysis

A comprehensive machine learning project analyzing mall customer data to identify distinct customer segments and predict customer behavior patterns.

## Team Members

- **Aleksandre Chakhvashvili** - Project Structure, ML Architecture, Model Implementation
- **Nino Gagnidze** - Data Cleaning, Exploratory Data Analysis, Documentation

## Project Overview

### Problem Statement

Understanding customer behavior is crucial for targeted marketing and business strategy. This project aims to segment mall customers based on their demographics and purchasing patterns to identify distinct customer groups and predict customer categories.

### Objectives

- Perform comprehensive data cleaning and preprocessing on mall customer data
- Conduct exploratory data analysis to uncover patterns and insights
- Implement K-Means clustering to segment customers into distinct groups
- Build classification models to predict customer segments
- Compare model performance and provide actionable business insights

### Dataset Description

**Source:** Mall Customers Dataset
**Records:** 200 customers
**Features:**

- CustomerID: Unique identifier for each customer
- Gender: Customer gender (Male/Female)
- Age: Customer age in years
- Annual Income (k$): Annual income in thousands of dollars
- Spending Score (1-100): Score assigned based on customer spending behavior and purchasing patterns

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/CHAKHVA/customer-segmentation-analysis.git
cd customer-segmentation-analysis
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Project Structure

```
customer-segmentation-analysis/
|-- data/
|   |-- raw/                      # Original mall_customers.csv dataset
|   +-- processed/                # Cleaned and transformed data
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_data_preprocessing.ipynb
|   |-- 03_eda_visualization.ipynb
|   +-- 04_machine_learning.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_processing.py        # Data cleaning functions
|   |-- visualization.py          # Plotting functions
|   +-- models.py                 # ML model implementations
|-- reports/
|   |-- figures/                  # Generated visualizations
|   +-- results/                  # Model outputs and metrics
|-- README.md
|-- CONTRIBUTIONS.md
|-- requirements.txt
+-- .gitignore
```

## Usage

Execute the notebooks in sequential order:

1. **01_data_exploration.ipynb** - Initial data inspection and quality assessment
2. **02_data_preprocessing.ipynb** - Data cleaning and feature engineering
3. **03_eda_visualization.ipynb** - Exploratory analysis and visualizations
4. **04_machine_learning.ipynb** - Model training and evaluation

## Methodology

### Data Processing

- Handling missing values and outliers
- Feature engineering and transformation
- Data normalization for clustering algorithms

### Exploratory Data Analysis

- Statistical analysis of customer demographics
- Distribution analysis of income and spending patterns
- Correlation analysis between features
- Visual exploration using multiple chart types

### Machine Learning Models

- **K-Means Clustering:** Unsupervised segmentation of customers
- **Classification Models:** Supervised prediction of customer segments

### Evaluation Metrics

- Clustering: Elbow method, Silhouette score, cluster visualization
- Classification: Accuracy, Precision, Recall, Confusion Matrix

## Results Summary

To be updated after analysis completion.

## Key Findings

To be updated after analysis completion.

## Dependencies

See `requirements.txt` for complete list of dependencies:

- numpy: Numerical computations
- pandas: Data manipulation and analysis
- matplotlib: Data visualization
- seaborn: Statistical visualizations
- scikit-learn: Machine learning algorithms
- jupyter: Interactive notebook environment

## Data Dictionary

To be updated with detailed feature descriptions after preprocessing.
