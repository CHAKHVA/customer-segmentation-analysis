"""
Machine Learning Models Module

This module contains functions for customer segmentation using clustering
and classification algorithms.

Author: Aleksandre Chakhvashvili
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    silhouette_score
)
from typing import Tuple, Dict, List, Optional, Any


def prepare_clustering_features(data: pd.DataFrame,
                                feature_columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prepare and scale features for clustering.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    feature_columns : List[str]
        List of feature columns to use for clustering

    Returns:
    --------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled features and the scaler object
    """
    X = data[feature_columns].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=data.index)

    return X_scaled_df, scaler


def find_optimal_clusters_elbow(data: pd.DataFrame,
                                max_k: int = 10) -> Dict[int, float]:
    """
    Find optimal number of clusters using the Elbow method.

    Parameters:
    -----------
    data : pd.DataFrame
        Scaled feature data
    max_k : int, optional
        Maximum number of clusters to test

    Returns:
    --------
    Dict[int, float]
        Dictionary mapping number of clusters to inertia (WCSS)
    """
    inertias = {}

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data)
        inertias[k] = kmeans.inertia_

    return inertias


def calculate_silhouette_scores(data: pd.DataFrame,
                                max_k: int = 10) -> Dict[int, float]:
    """
    Calculate silhouette scores for different numbers of clusters.

    Parameters:
    -----------
    data : pd.DataFrame
        Scaled feature data
    max_k : int, optional
        Maximum number of clusters to test

    Returns:
    --------
    Dict[int, float]
        Dictionary mapping number of clusters to silhouette score
    """
    silhouette_scores = {}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores[k] = silhouette_avg

    return silhouette_scores


def plot_elbow_curve(inertias: Dict[int, float],
                    save_path: Optional[str] = None) -> None:
    """
    Plot elbow curve for K-Means clustering.

    Parameters:
    -----------
    inertias : Dict[int, float]
        Dictionary mapping number of clusters to inertia
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(inertias.keys()), list(inertias.values()),
             marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')

    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(inertias.keys()))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Elbow curve saved to: {save_path}")

    plt.show()


def plot_silhouette_scores(silhouette_scores: Dict[int, float],
                          save_path: Optional[str] = None) -> None:
    """
    Plot silhouette scores for different numbers of clusters.

    Parameters:
    -----------
    silhouette_scores : Dict[int, float]
        Dictionary mapping number of clusters to silhouette score
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()),
             marker='o', linestyle='-', linewidth=2, markersize=8, color='green')

    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score for Different k Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(silhouette_scores.keys()))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Silhouette scores plot saved to: {save_path}")

    plt.show()


def train_kmeans(data: pd.DataFrame,
                n_clusters: int,
                random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    """
    Train K-Means clustering model.

    Parameters:
    -----------
    data : pd.DataFrame
        Scaled feature data
    n_clusters : int
        Number of clusters
    random_state : int, optional
        Random state for reproducibility

    Returns:
    --------
    Tuple[KMeans, np.ndarray]
        Trained KMeans model and cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                   random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    print(f"K-Means clustering completed with {n_clusters} clusters.")
    print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")

    # Calculate silhouette score
    sil_score = silhouette_score(data, cluster_labels)
    print(f"Silhouette Score: {sil_score:.3f}")

    return kmeans, cluster_labels


def visualize_clusters_2d(data_original: pd.DataFrame,
                          cluster_labels: np.ndarray,
                          x_col: str,
                          y_col: str,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Visualize clusters in 2D space.

    Parameters:
    -----------
    data_original : pd.DataFrame
        Original (unscaled) data
    cluster_labels : np.ndarray
        Cluster labels
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 8))

    # Create a copy with cluster labels
    plot_data = data_original.copy()
    plot_data['Cluster'] = cluster_labels

    # Plot each cluster
    unique_clusters = sorted(plot_data['Cluster'].unique())
    colors = sns.color_palette('Set2', len(unique_clusters))

    for idx, cluster in enumerate(unique_clusters):
        cluster_data = plot_data[plot_data['Cluster'] == cluster]
        plt.scatter(cluster_data[x_col], cluster_data[y_col],
                   s=100, alpha=0.7, c=[colors[idx]], label=f'Cluster {cluster}',
                   edgecolor='black', linewidth=0.5)

    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title or f'Customer Segments - {y_col} vs {x_col}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster visualization saved to: {save_path}")

    plt.show()


def analyze_clusters(data_original: pd.DataFrame,
                    cluster_labels: np.ndarray,
                    feature_columns: List[str]) -> pd.DataFrame:
    """
    Analyze cluster characteristics.

    Parameters:
    -----------
    data_original : pd.DataFrame
        Original data
    cluster_labels : np.ndarray
        Cluster labels
    feature_columns : List[str]
        List of features to analyze

    Returns:
    --------
    pd.DataFrame
        Cluster analysis summary
    """
    analysis_data = data_original.copy()
    analysis_data['Cluster'] = cluster_labels

    # Calculate statistics for each cluster
    cluster_summary = analysis_data.groupby('Cluster')[feature_columns].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    return cluster_summary


def describe_clusters(data_original: pd.DataFrame,
                     cluster_labels: np.ndarray) -> Dict[int, str]:
    """
    Generate descriptive names/characteristics for each cluster.

    Parameters:
    -----------
    data_original : pd.DataFrame
        Original data with Income and Spending Score
    cluster_labels : np.ndarray
        Cluster labels

    Returns:
    --------
    Dict[int, str]
        Dictionary mapping cluster number to description
    """
    analysis_data = data_original.copy()
    analysis_data['Cluster'] = cluster_labels

    cluster_descriptions = {}

    for cluster in sorted(analysis_data['Cluster'].unique()):
        cluster_data = analysis_data[analysis_data['Cluster'] == cluster]

        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        size = len(cluster_data)

        # Classify income level
        if avg_income < 40:
            income_level = "Low Income"
        elif avg_income < 70:
            income_level = "Medium Income"
        else:
            income_level = "High Income"

        # Classify spending level
        if avg_spending < 35:
            spending_level = "Low Spender"
        elif avg_spending < 65:
            spending_level = "Medium Spender"
        else:
            spending_level = "High Spender"

        description = f"{income_level}, {spending_level} (n={size})"
        cluster_descriptions[cluster] = description

    return cluster_descriptions


def prepare_classification_data(data: pd.DataFrame,
                               feature_columns: List[str],
                               target_column: str,
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple:
    """
    Prepare data for classification.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    feature_columns : List[str]
        List of feature columns
    target_column : str
        Target column name
    test_size : float, optional
        Proportion of test set
    random_state : int, optional
        Random state for reproducibility

    Returns:
    --------
    Tuple
        X_train, X_test, y_train, y_test
    """
    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Target distribution in training set:\n{y_train.value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              random_state: int = 42) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int, optional
        Random state for reproducibility

    Returns:
    --------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    print("Logistic Regression model trained successfully.")
    return model


def train_decision_tree(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train Decision Tree classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int, optional
        Random state for reproducibility

    Returns:
    --------
    DecisionTreeClassifier
        Trained model
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    print("Decision Tree model trained successfully.")
    return model


def evaluate_classifier(model: Any,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model_name: str = "Model") -> Dict[str, Any]:
    """
    Evaluate classification model.

    Parameters:
    -----------
    model : Any
        Trained classifier
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    model_name : str, optional
        Name of the model for display

    Returns:
    --------
    Dict[str, Any]
        Evaluation metrics
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} Evaluation Results:")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }


def plot_confusion_matrix(conf_matrix: np.ndarray,
                         model_name: str,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix heatmap.

    Parameters:
    -----------
    conf_matrix : np.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               square=True, cbar_kws={"shrink": 0.8})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Dictionary of model results

    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison_data = []

    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)

    return comparison_df
