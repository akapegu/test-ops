"""
Inference pipeline for Housing Regression MLE.

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing + feature engineering using saved encoders.
- Aligns features with training.
- Returns predictions.
"""

# Raw → preprocess → feature engineering → align schema → model.predict → predictions.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    # Step 1: Preprocess raw input
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Feature engineering
    if "date" in df.columns:
        df = add_date_features(df)

    # Step 3: Encodings ----------------
    # Frequency encoding (zipcode)
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    # Target encoding (city_full → city_full_encoded)
    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full"], errors="ignore")

    # Drop leakage columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Backward compat: notebook data used city_encoded, pipeline uses city_full_encoded
    if "city_encoded" in df.columns and "city_full_encoded" not in df.columns:
        df["city_full_encoded"] = df.pop("city_encoded")

    # Step 4: Separate actuals if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Step 5: Load model & align columns to its training schema
    model = load(model_path)
    feature_cols = list(model.feature_names_in_)
    df = df.reindex(columns=feature_cols, fill_value=0)
    # JSON nulls can create object-dtype columns; XGBoost requires numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Step 6: Predict
    preds = model.predict(df)

    # Step 7: Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER), help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str, default=str(DEFAULT_TARGET_ENCODER), help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
