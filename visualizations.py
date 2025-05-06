import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns   
import numpy as np 



def na_visualization(df,save=False): 
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False)
    plt.title("Map of missing values ")
    plt.tight_layout()
    plt.show()

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if not missing_counts.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=missing_counts.index, y=missing_counts.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Number of missing values")
        plt.title("Missing Values by Column")
        plt.tight_layout()
        plt.show()
    else:
        print("There are no missing values")


def correlation_heatmap(df, save=False):
    numeric_cols = df.select_dtypes("number").columns
    corr = df[numeric_cols].corr(method="pearson")
    
    # Imprime la tabla de correlación en consola
    print("\nTabla de correlaciones:")
    print(corr.round(2))  # Redondea a 2 decimales para que sea más legible
    
    # Dibuja el heatmap
    plt.figure(figsize=(14, 12))
    sns.set(font_scale=0.8)
    
    sns.heatmap(corr, annot=False, fmt=".1f", cmap="coolwarm", square=True,
                cbar_kws={'shrink':0.8})
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.title("Correlation Matrix ")
    plt.tight_layout()
    
    if save:
        plt.savefig('correlation_heatmap.png', dpi=300)
    plt.show()

def visualize_numeric_cols(df):
    numeric_cols = df.select_dtypes("number").columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col])
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        # plt.savefig(PLOTS_DIR / f"dist_{col}.png")
        # plt.close()

def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 20,
    save: bool = False,
    figsize: tuple[int, int] = (8, 10),
):
    """Horizontal bar chart of the *top_n* most important features."""
    top = importance_df.head(top_n)[::-1]  # reverse for ascending y‑axis
    plt.figure(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=top, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save:
        plt.savefig("feature_importance_top.png", dpi=300)
    plt.show()


def feature_importance_heatmap(
    importance_df: pd.DataFrame,
    *,
    save: bool = False,
    figsize: tuple[int, int] = (16, 3),
):
    """Heatmap of all feature importances for a global overview."""
    plt.figure(figsize=figsize)
    # Convert to matrix: one row, features as columns
    data = importance_df.set_index("feature").T
    sns.heatmap(data, cmap="viridis", cbar_kws={"label": "Importance"})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title("Feature Importances – Global Heatmap")
    plt.tight_layout()
    if save:
        plt.savefig("feature_importance_heatmap.png", dpi=300)
    plt.show()

def plot_pca_variance(
    explained_ratio: np.ndarray,
    *,
    save: bool = False,
    figsize: tuple[int, int] = (10, 5),
):
    """Plot scree plot and cumulative variance curve from PCA explained_ratio."""
    cum_var = np.cumsum(explained_ratio)
    comps = np.arange(1, len(explained_ratio) + 1)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(comps, explained_ratio, alpha=0.6, label="Individual variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")

    ax2 = ax1.twinx()
    ax2.plot(comps, cum_var, color="red", marker="o", label="Cumulative variance")
    ax2.set_ylabel("Cumulative Variance Ratio")
    ax2.grid(False)

    ax1.set_title("PCA – Explained Variance per Component")

    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.tight_layout()
    if save:
        plt.savefig("pca_variance.png", dpi=300)
    plt.show()