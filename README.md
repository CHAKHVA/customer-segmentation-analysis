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

### Dataset Statistics

- Total customers analyzed: 200
- Features in final dataset: 9 (5 original + 4 engineered)
- Data quality: No missing values, no duplicates
- Outliers: 2 high-income customers (kept as valid segments)

### K-Means Clustering Results

- Optimal number of clusters: 5
- Silhouette Score: 0.5547 (indicating good cluster separation)
- Features used: Annual Income and Spending Score
- WCSS (Inertia): 65.57

### Customer Segments Identified

1. **Cluster 0: Medium Income, Medium Spender** (n=81, 40.5%)
   - Average Income: $55.3k
   - Average Spending Score: 49.5
   - Largest segment, balanced purchasing behavior

2. **Cluster 1: High Income, High Spender** (n=39, 19.5%)
   - Average Income: $86.5k
   - Average Spending Score: 82.1
   - Premium customers, high value targets

3. **Cluster 2: Low Income, High Spender** (n=22, 11.0%)
   - Average Income: $25.7k
   - Average Spending Score: 79.4
   - Impulsive buyers despite limited income

4. **Cluster 3: High Income, Low Spender** (n=35, 17.5%)
   - Average Income: $88.2k
   - Average Spending Score: 17.1
   - Conservative spenders, potential growth opportunity

5. **Cluster 4: Low Income, Low Spender** (n=23, 11.5%)
   - Average Income: $26.3k
   - Average Spending Score: 20.9
   - Budget-conscious segment

### Classification Model Performance

Two models were trained to predict customer segments:

| Model               | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Logistic Regression | 97.50%   | 97.65%    | 97.50% |
| Decision Tree       | 95.00%   | 95.56%    | 95.00% |

**Best Model:** Logistic Regression (97.5% accuracy)

The high accuracy indicates that customer segments can be reliably predicted using demographic and behavioral features.

### Visualizations Created

- 25 total visualizations across all notebooks
- 8 different chart types: distributions, box plots, correlation heatmaps, scatter plots, bar charts, pair plots, violin plots, pie charts
- All figures saved in `reports/figures/`

## Key Findings

### 1. Customer Demographics

- Gender distribution: 56% Female, 44% Male
- Average customer age: 38.9 years
- Age groups: Middle-Aged (31%), Adult (30%), Senior (20%), Young (19%)
- Average annual income: $60.6k

### 2. Spending Patterns

- Average spending score: 50.2 (on 1-100 scale)
- Spending distribution: Medium (43.5%), High (29%), Low (27.5%)
- Key insight: Age negatively correlates with spending score (-0.33)
  - Younger customers tend to spend more
  - Older customers are more conservative with spending

### 3. Income vs Spending Relationship

- Correlation between income and spending: 0.01 (nearly zero)
- Key insight: Income level does NOT directly predict spending behavior
- This validates the need for clustering to identify distinct segments

### 4. Gender Differences

- Minimal correlation between gender and spending (-0.06)
- Female customers: Average spending score 51.5
- Male customers: Average spending score 48.5
- Gender is not a strong predictor of spending behavior

### 5. Business Recommendations

**For Cluster 1 (High Income, High Spenders):**

- Premium product offerings and exclusive memberships
- Personalized shopping experiences
- Loyalty rewards programs

**For Cluster 2 (Low Income, High Spenders):**

- Installment payment options
- Credit facilities and buy-now-pay-later schemes
- Frequent promotional offers

**For Cluster 3 (High Income, Low Spenders):**

- Targeted marketing to increase engagement
- Quality-focused messaging over discounts
- Exclusive events to drive store visits

**For Cluster 0 (Medium Income, Medium Spenders):**

- Balanced product range
- Seasonal promotions
- Value-for-money positioning

**For Cluster 4 (Low Income, Low Spenders):**

- Budget-friendly product lines
- Clearance sales and discounts
- Entry-level offerings

## Dependencies

See `requirements.txt` for complete list of dependencies:

- numpy: Numerical computations
- pandas: Data manipulation and analysis
- matplotlib: Data visualization
- seaborn: Statistical visualizations
- scikit-learn: Machine learning algorithms
- jupyter: Interactive notebook environment

## Data Dictionary

### Original Features

| Feature                | Type        | Description                                 | Range/Values |
| ---------------------- | ----------- | ------------------------------------------- | ------------ |
| CustomerID             | Integer     | Unique identifier for each customer         | 1-200        |
| Gender                 | Categorical | Customer gender                             | Male, Female |
| Age                    | Integer     | Customer age in years                       | 18-70        |
| Annual Income (k$)     | Integer     | Annual income in thousands of dollars       | 15-137       |
| Spending Score (1-100) | Integer     | Score assigned based on purchasing behavior | 1-99         |

### Engineered Features

| Feature           | Type        | Description                      | Categories                                                        |
| ----------------- | ----------- | -------------------------------- | ----------------------------------------------------------------- |
| Age_Group         | Categorical | Age range classification         | Young (18-25), Adult (26-35), Middle-Aged (36-50), Senior (50+)   |
| Income_Category   | Categorical | Income level classification      | Low (<40k), Medium (40-70k), High (70-100k), Very High (>100k)    |
| Spending_Category | Categorical | Spending behavior classification | Low Spender (1-35), Medium Spender (36-65), High Spender (66-100) |
| Gender_Encoded    | Binary      | Numerical encoding of gender     | 0 (Female), 1 (Male)                                              |
| Cluster           | Integer     | Customer segment from K-Means    | 0-4                                                               |

### Statistical Summary

| Feature                | Mean  | Median | Std Dev | Min | Max |
| ---------------------- | ----- | ------ | ------- | --- | --- |
| Age                    | 38.85 | 36.0   | 13.97   | 18  | 70  |
| Annual Income (k$)     | 60.56 | 61.5   | 26.26   | 15  | 137 |
| Spending Score (1-100) | 50.20 | 50.0   | 25.82   | 1   | 99  |

## Project Outcomes

### Technical Achievements

- Successfully implemented end-to-end machine learning pipeline
- Achieved 97.5% classification accuracy with Logistic Regression
- Identified 5 distinct customer segments with clear business value
- Created 25 publication-quality visualizations
- Developed reusable Python modules for future analysis

### Course Requirements Met

- Data Processing: Comprehensive cleaning with Pandas, outlier detection, feature engineering
- EDA: 8+ visualization types, statistical analysis, correlation studies
- ML Models: K-Means clustering + 2 classification models (Logistic Regression, Decision Tree)
- Code Quality: Well-documented, modular code with proper error handling
- Evaluation: Accuracy, precision, recall, confusion matrices, silhouette score

### Business Impact

The analysis provides actionable insights for:

- Targeted marketing strategies for each customer segment
- Product positioning and pricing optimization
- Resource allocation for customer acquisition and retention
- Personalized customer experience initiatives
