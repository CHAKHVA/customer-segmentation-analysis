"""
Visualization Module

This module contains reusable plotting functions for exploratory data analysis.

Author: Nino Gagnidze
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_distribution(data: pd.DataFrame,
                     column: str,
                     title: Optional[str] = None,
                     bins: int = 30,
                     color: str = 'skyblue',
                     save_path: Optional[str] = None) -> None:
    """
    Plot distribution of a numerical variable using histogram with KDE.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    title : str, optional
        Plot title
    bins : int, optional
        Number of bins for histogram
    color : str, optional
        Color for the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))

    # Create histogram with KDE
    plt.hist(data[column], bins=bins, color=color, alpha=0.7, edgecolor='black', density=True)

    # Add KDE
    data[column].plot(kind='kde', color='darkblue', linewidth=2)

    # Add mean and median lines
    mean_val = data[column].mean()
    median_val = data[column].median()

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_boxplot(data: pd.DataFrame,
                columns: List[str],
                title: Optional[str] = None,
                save_path: Optional[str] = None) -> None:
    """
    Plot box plots for multiple numerical variables.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of column names to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(6*len(columns), 6))

    if len(columns) == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        sns.boxplot(y=data[col], ax=axes[idx], color='lightblue')
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(title or 'Box Plots of Numerical Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_correlation_heatmap(data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap for numerical features.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        List of columns to include in correlation
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    if columns:
        corr_data = data[columns].corr()
    else:
        corr_data = data.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title(title or 'Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_scatter(data: pd.DataFrame,
                x_col: str,
                y_col: str,
                hue: Optional[str] = None,
                title: Optional[str] = None,
                save_path: Optional[str] = None) -> None:
    """
    Plot scatter plot for two numerical variables.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    hue : str, optional
        Column name for color coding
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))

    if hue:
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, s=100, alpha=0.7)
    else:
        sns.scatterplot(data=data, x=x_col, y=y_col, s=100, alpha=0.7, color='steelblue')

    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title or f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_count_bar(data: pd.DataFrame,
                  column: str,
                  title: Optional[str] = None,
                  horizontal: bool = False,
                  save_path: Optional[str] = None) -> None:
    """
    Plot count bar chart for categorical variable.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    title : str, optional
        Plot title
    horizontal : bool, optional
        Whether to plot horizontal bars
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))

    counts = data[column].value_counts()

    if horizontal:
        counts.plot(kind='barh', color='steelblue', edgecolor='black')
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(column, fontsize=12)
    else:
        counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    if horizontal:
        for i, v in enumerate(counts):
            plt.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=10)
    else:
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts)*0.01, str(v), ha='center', fontsize=10)

    plt.title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_pairplot(data: pd.DataFrame,
                 columns: List[str],
                 hue: Optional[str] = None,
                 save_path: Optional[str] = None) -> None:
    """
    Plot pairplot for multiple numerical variables.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to include
    hue : str, optional
        Column name for color coding
    save_path : str, optional
        Path to save the figure
    """
    if hue and hue not in columns:
        plot_columns = columns + [hue]
    else:
        plot_columns = columns

    pairplot = sns.pairplot(data[plot_columns], hue=hue, diag_kind='kde',
                           plot_kws={'alpha': 0.6, 's': 80}, height=2.5)
    pairplot.fig.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_grouped_bar(data: pd.DataFrame,
                    x_col: str,
                    hue_col: str,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Plot grouped bar chart.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    x_col : str
        Column for x-axis (categorical)
    hue_col : str
        Column for grouping (categorical)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))

    # Create crosstab
    ct = pd.crosstab(data[x_col], data[hue_col])
    ct.plot(kind='bar', edgecolor='black')

    plt.xlabel(x_col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title or f'{x_col} by {hue_col}', fontsize=14, fontweight='bold')
    plt.legend(title=hue_col, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_violin(data: pd.DataFrame,
               x_col: Optional[str] = None,
               y_col: str = None,
               title: Optional[str] = None,
               save_path: Optional[str] = None) -> None:
    """
    Plot violin plot.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    x_col : str, optional
        Column for x-axis (categorical)
    y_col : str
        Column for y-axis (numerical)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))

    if x_col:
        sns.violinplot(data=data, x=x_col, y=y_col, palette='Set2')
        plt.xticks(rotation=45, ha='right')
    else:
        sns.violinplot(data=data, y=y_col, color='lightblue')

    plt.xlabel(x_col if x_col else '', fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title or f'Violin Plot of {y_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_pie_chart(data: pd.DataFrame,
                  column: str,
                  title: Optional[str] = None,
                  save_path: Optional[str] = None) -> None:
    """
    Plot pie chart for categorical variable.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(8, 8))

    counts = data[column].value_counts()
    colors = sns.color_palette('Set2', len(counts))

    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 11})

    plt.title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_multiple_distributions(data: pd.DataFrame,
                               columns: List[str],
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Plot multiple distribution plots in a grid.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of column names to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row

    fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(columns):
        axes[idx].hist(data[col], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title or 'Distribution of Numerical Features', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def create_statistical_summary_table(data: pd.DataFrame,
                                     columns: List[str]) -> pd.DataFrame:
    """
    Create a comprehensive statistical summary table.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to analyze

    Returns:
    --------
    pd.DataFrame
        Statistical summary dataframe
    """
    summary = pd.DataFrame({
        'Mean': data[columns].mean(),
        'Median': data[columns].median(),
        'Std': data[columns].std(),
        'Min': data[columns].min(),
        'Max': data[columns].max(),
        'Q1': data[columns].quantile(0.25),
        'Q3': data[columns].quantile(0.75),
        'IQR': data[columns].quantile(0.75) - data[columns].quantile(0.25),
        'Skewness': data[columns].skew(),
        'Kurtosis': data[columns].kurtosis()
    })

    return summary.round(2)
