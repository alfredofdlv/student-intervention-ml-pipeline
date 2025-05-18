
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import scipy.stats as ss


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> List[int]:
    numeric_cols = df.select_dtypes("number").columns
    outlier_idx: List[int] = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_idx.extend(df[mask].index.tolist())
    outlier_idx = sorted(set(outlier_idx))
    print(f"Detected {len(outlier_idx)} potential outliers across numeric features.")
    return outlier_idx

def build_features(X,columns=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'] )-> pd.DataFrame:

    """Feature engineering after doing the preprocessing ."""
    if not isinstance(X, pd.DataFrame):
        df = pd.DataFrame(X, columns=columns)
    else:
        df = X.copy()
    # 1. Leisure–study balance
    df['leisure_balance'] = df['freetime'] - df['studytime']
    
    # 2. Aggregate social activities
    df['sociality_index'] = (
        df['goout'] +
        df['romantic'] +
        df['activities'] +
        df['nursery']
    )
    
    # 3. Alcohol consumption index
    df['alcohol_index'] = 0.4 * df['Dalc'] + 0.6 * df['Walc']
    
    # 4. Health risk score
    df['health_risk_score'] = (6 - df['health']) * df['alcohol_index']
    
    # 5. Digital access
    df['tech_access'] = df['internet'] + df['address']
    
    # 6. Total formal support
    df['support_sum'] = (
        df['schoolsup'] +
        df['famsup'] +
        df['paid']
    )
    
    # 7. Support mismatch
    df['support_mismatch'] = df['schoolsup'] - df['famsup']
    
    # 8. Absence ratio adjusted by commute time
    df['absence_ratio'] = df['absences'] / (df['traveltime'] + 1)
    
    # 9. Long commuter flag
    df['long_commuter_flag'] = (df['traveltime'] >= 3).astype(int)
    return df 

    

def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model: RandomForestClassifier | None = None,
    random_state: int = 42,
    n_estimators: int = 400,
) -> tuple[pd.DataFrame, RandomForestClassifier]:
    """Train *model* (defaults to RandomForest) and return a DataFrame of feature importances sorted descending."""
    if model is None:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    model.fit(X, y)
    imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    imp.sort_values("importance", ascending=False, inplace=True)
    imp.reset_index(drop=True, inplace=True)
    return imp, model

def preprocess_and_pca(X_train, X_test, show_plot=True):
    """
    Realiza PCA en el conjunto de datos.

    Args:
        X_train (DataFrame): Datos de entrenamiento.
        X_test (DataFrame): Datos de prueba.
        show_plot (bool): Si True, muestra un gráfico de varianza explicada acumulada.

    Returns:
        tuple: X_train_pca, X_test_pca, pca
    """
    pca = PCA(n_components=0.975)
    pca.fit(X_train)

    # Autovalores y criterios de Kaiser
    autovalores = pca.explained_variance_
    media_autovalores = np.mean(autovalores)

    criterio_kaiser_cov = autovalores > media_autovalores
    criterio_kaiser_cor = autovalores > 1

    # Varianza explicada y acumulada
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    # Impresiones
    print(f"Components (Kaiser criterion - covariance > mean): {np.sum(criterio_kaiser_cov)}")
    print(f"  Cumulative variance: {varianza_acumulada[np.sum(criterio_kaiser_cov) - 1]:.4f}")

    print(f"Components (Kaiser criterion - correlation > 1): {np.sum(criterio_kaiser_cor)}")
    print(f"  Cumulative variance: {varianza_acumulada[np.sum(criterio_kaiser_cor) - 1]:.4f}")

    # Print number of components selected to reach 97.5% variance
    n_components_975 = pca.n_components_
    print(f"Number of components to reach 97.5% variance: {n_components_975}")

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca
def cramers_v(x, y):
        cm = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(cm)[0]
        n = cm.values.sum()
        return np.sqrt(chi2 / (n * (min(cm.shape) - 1)))
 
