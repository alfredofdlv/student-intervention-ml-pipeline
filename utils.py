
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


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

def build_features(df_part: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    """Añade 16 nuevas columnas al DataFrame usando `stats` pre‑calculadas."""
    out = df_part.copy()

    # --- Imputaciones internas para evitar NaNs en features derivados ---
    out["studytime_fill"] = out["studytime"].fillna(stats["median_studytime"])
    out["absences_fill"] = out["absences"].fillna(0)
    out["traveltime_fill"] = out["traveltime"].fillna(stats["median_traveltime"])


    # 1) study_efficiency_log
    out["study_efficiency_log"] = np.log(out["studytime_fill"] + 1) - np.log(out["absences_fill"] + 1)

    # 2) absence_ratio
    out["absence_ratio"] = out["absences"] / (out["traveltime"] + 1)

    # 3) absence_flag_q75
    out["absence_flag_q75"] = (out["absences"] > stats["q75_absences"]).astype(int)

    # 4) alcohol_index_z
    z_dalc = (out["Dalc"] - stats["mean_dalc"]) / stats["std_dalc"]
    z_walc = (out["Walc"] - stats["mean_walc"]) / stats["std_walc"]
    out["alcohol_index_z"] = 0.4 * z_dalc + 0.6 * z_walc

    # 5) alcohol_spike_weekend
    out["alcohol_spike_weekend"] = out["Walc"] - out["Dalc"]

    # 6) weekend_focus
    out["weekend_focus"] = ((out["Dalc"] <= 2) & (out["Walc"] <= 2)).astype(int)

    # 7) family_support_total
    out["family_support_total"] = out["schoolsup_yes"] + out["famsup_yes"] + out["paid_yes"]

    # 8) support_change
    out["support_change"] = out["schoolsup_yes"] - out["famsup_yes"]

    # 9) parent_edu_avg
    out["parent_edu_avg"] = (out["Medu"] + out["Fedu"]) / 2

    # 10) highly_educated_parent
    out["highly_educated_parent"] = (np.maximum(out["Medu"], out["Fedu"]) >= 3).astype(int)

    # 11‑12) commute spline
    out["commute_spline_low"] = np.maximum(0, 2 - out["traveltime"])
    out["commute_spline_high"] = np.maximum(0, out["traveltime"] - 2)

    # 13) failure_history_std
    out["failure_history_std"] = (out["failures"] - stats["mean_failures"]) / stats["std_failures"]

    # 14) failure_history_sq
    out["failure_history_sq"] = out["failure_history_std"] ** 2

    # 15‑16) fairness interactions
    out["fairness_sex_schoolsup"] = out["sex_F"] * out["schoolsup_yes"]
    out["fairness_sex_famsup"] = out["sex_F"] * out["famsup_yes"]

    # Eliminar helpers *_fill para no duplicar columnas
    
    out['study_efficiency_log']=out['study_efficiency_log'].fillna(0)
    # print(out['study_efficiency_log'])
    out.drop(columns=["studytime_fill", "absences_fill", "traveltime_fill",'study_efficiency_log'], inplace=True)

    return out

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