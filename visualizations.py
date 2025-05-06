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