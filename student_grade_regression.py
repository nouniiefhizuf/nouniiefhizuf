import os
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


def download_dataset() -> str:
    try:
        import kagglehub
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        print("kagglehub is required. Install dependencies from requirements.txt", file=sys.stderr)
        raise exc

    path = kagglehub.dataset_download("tejas14/student-final-grade-prediction-multi-lin-reg")
    print("Path to dataset files:", path)
    return path


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    csv_files: List[Path] = list(Path(dataset_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    # Combine if multiple CSVs exist; otherwise just read the single file
    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Normalize common column names if needed
    # Expecting target column to be 'G3' (final grade) as in UCI/Kaggle datasets
    return data


def preprocess_and_split(data: pd.DataFrame, target_column: str = "G3", test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    if target_column not in data.columns:
        # Some notebooks rename target to 'final_grade' or similar; attempt to infer
        candidates = [c for c in data.columns if c.lower() in {"g3", "final_grade", "finalgrade", "final"}]
        if not candidates:
            raise KeyError(f"Target column '{target_column}' not found and could not infer from {list(data.columns)}")
        target_column = candidates[0]

    X = data.drop(columns=[target_column])
    y = data[target_column].values

    numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_models(alphas: List[float] | None = None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 25)

    lr = LinearRegression(n_jobs=None)
    ridge = RidgeCV(alphas=alphas, store_cv_values=False)
    lasso = LassoCV(alphas=alphas, max_iter=5000, n_jobs=None)
    return {
        "LinearRegression": lr,
        "RidgeCV": ridge,
        "LassoCV": lasso,
    }


def evaluate_and_plot(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    # Residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals, s=25)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Final Grade")
    plt.ylabel("Residuals (True - Pred)")
    plt.title(f"Residuals: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"residuals_{model_name}.png", dpi=150)
    plt.close()

    # Parity plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=25)
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)
    plt.xlabel("True Final Grade")
    plt.ylabel("Predicted Final Grade")
    plt.title(f"Parity: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"parity_{model_name}.png", dpi=150)
    plt.close()

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = download_dataset()
    df = load_dataset(dataset_dir)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

    models = build_models()

    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ])

        # Cross-validation on training set
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        neg_rmse_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=None,
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_and_plot(name, y_test, y_pred, artifacts_dir)
        metrics["CV_RMSE_Mean"] = float(-np.mean(neg_rmse_scores))
        metrics["CV_RMSE_Std"] = float(np.std(-neg_rmse_scores))
        results[name] = metrics

        # Persist pipeline
        joblib.dump(pipeline, artifacts_dir / f"model_{name}.joblib")

    # Save metrics
    results_df = pd.DataFrame(results).T
    results_df.to_csv(artifacts_dir / "metrics.csv", index=True)
    print("\nModel performance (test set):")
    print(results_df.round(4))


if __name__ == "__main__":
    main()

