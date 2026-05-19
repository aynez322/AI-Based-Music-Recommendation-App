import json
import os
import pickle
from typing import Optional, Type

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_based_music_recommendation.tools.dataset_tools import (
    MERGED_PATH,
    NUMERIC_FEATURES,
    _find_index,
    _load,
)

_DATA_DIR   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
_MODEL_PATH = os.path.join(_DATA_DIR, "genre_classifier.pkl")

_model: Optional[xgb.XGBClassifier] = None
_encoder: Optional[LabelEncoder] = None
_explainer: Optional[shap.TreeExplainer] = None


def _load_training_data() -> pd.DataFrame:
    """Return labeled rows from merged_dataset.parquet for XGBoost training."""
    if os.path.exists(MERGED_PATH):
        print("  Using labeled rows from merged_dataset.parquet as training source")
        df = pd.read_parquet(MERGED_PATH)
        df = df[df["track_genre"] != "unknown"].dropna(subset=NUMERIC_FEATURES + ["track_genre"])
        if len(df) < 1000:
            raise RuntimeError(
                "Not enough labeled songs in merged_dataset.parquet to train the genre "
                "classifier (need at least 1 000). Run merge_datasets.py first and ensure "
                "at least one dataset with genre labels is included."
            )
        return df

    raise FileNotFoundError(
        "No training data found for the genre classifier.\n"
        "Either place maharshipandya/dataset.csv in data/ or run merge_datasets.py "
        "with at least one genre-labeled dataset."
    )


def _get_model_and_explainer() -> tuple:
    global _model, _encoder, _explainer
    if _model is not None:
        return _model, _encoder, _explainer

    if os.path.exists(_MODEL_PATH):
        print("Loading cached genre classifier...")
        with open(_MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        _model = saved["model"]
        _encoder = saved["encoder"]
    else:
        print("Training XGBoost genre classifier (first run only — may take 1-2 minutes)...")
        df_train = _load_training_data()
        print(f"  Training on {len(df_train):,} labeled songs, {df_train['track_genre'].nunique()} genres")

        le = LabelEncoder()
        y = le.fit_transform(df_train["track_genre"])
        X = df_train[NUMERIC_FEATURES].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        acc = model.score(X_test, y_test)
        print(f"  Genre classifier accuracy: {acc:.3f}")

        with open(_MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "encoder": le}, f)
        _model = model
        _encoder = le

    _explainer = shap.TreeExplainer(_model)
    return _model, _encoder, _explainer


def _shap_for_class(explainer: shap.TreeExplainer, X_row: np.ndarray, pred_class: int) -> np.ndarray:
    try:
        exp = explainer(X_row, check_additivity=False)
        vals = exp.values
        if vals.ndim == 3:
            return vals[0, :, pred_class]
        return vals[0]
    except Exception:
        sv = explainer.shap_values(X_row, check_additivity=False)
        if isinstance(sv, list):
            return sv[pred_class][0]
        if sv.ndim == 3:
            return sv[0, :, pred_class]
        return sv[0]


class SHAPExplainInput(BaseModel):
    input_song: str = Field(..., description="Title of the input/reference song")
    recommended_song: str = Field(..., description="Title of the recommended song to explain")
    input_artist: str = Field(default="", description="Artist of the input song (optional)")
    recommended_artist: str = Field(default="", description="Artist of the recommended song (optional)")


class SHAPExplainerTool(BaseTool):
    name: str = "shap_explain_similarity"
    description: str = (
        "Explains why a recommended song is similar to the input song using a trained "
        "XGBoost genre classifier and SHAP (SHapley Additive exPlanations). "
        "Reveals which audio features (energy, valence, tempo, acousticness, etc.) are "
        "the main drivers of similarity between the two songs."
    )
    args_schema: Type[BaseModel] = SHAPExplainInput

    def _run(
        self,
        input_song: str,
        recommended_song: str,
        input_artist: str = "",
        recommended_artist: str = "",
    ) -> str:
        try:
            df, _ = _load()
            model, le, explainer = _get_model_and_explainer()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        in_idx = _find_index(df, input_song, input_artist)
        rec_idx = _find_index(df, recommended_song, recommended_artist)

        if in_idx is None:
            return json.dumps({"error": f"'{input_song}' not found in dataset."})
        if rec_idx is None:
            return json.dumps({"error": f"'{recommended_song}' not found in dataset."})

        in_X = df.loc[in_idx, NUMERIC_FEATURES].values.astype(float).reshape(1, -1)
        rec_X = df.loc[rec_idx, NUMERIC_FEATURES].values.astype(float).reshape(1, -1)

        in_pred = int(model.predict(in_X)[0])
        rec_pred = int(model.predict(rec_X)[0])
        in_genre_pred = le.inverse_transform([in_pred])[0]
        rec_genre_pred = le.inverse_transform([rec_pred])[0]

        in_shap = _shap_for_class(explainer, in_X, in_pred)
        rec_shap = _shap_for_class(explainer, rec_X, rec_pred)

        in_vals = {f: round(float(df.loc[in_idx, f]), 4) for f in NUMERIC_FEATURES}
        rec_vals = {f: round(float(df.loc[rec_idx, f]), 4) for f in NUMERIC_FEATURES}

        features = []
        for i, feat in enumerate(NUMERIC_FEATURES):
            sv_in = float(in_shap[i])
            sv_rec = float(rec_shap[i])
            aligned = (sv_in >= 0) == (sv_rec >= 0)
            features.append(
                {
                    "feature": feat,
                    "input_value": in_vals[feat],
                    "recommended_value": rec_vals[feat],
                    "input_shap": round(sv_in, 4),
                    "recommended_shap": round(sv_rec, 4),
                    "aligned": aligned,
                    "avg_abs_shap": round((abs(sv_in) + abs(sv_rec)) / 2, 4),
                }
            )

        features.sort(key=lambda x: x["avg_abs_shap"], reverse=True)
        top_aligned = [f for f in features if f["aligned"]][:4]

        return json.dumps(
            {
                "input_song": str(df.loc[in_idx, "track_name"]),
                "recommended_song": str(df.loc[rec_idx, "track_name"]),
                "input_predicted_genre": in_genre_pred,
                "recommended_predicted_genre": rec_genre_pred,
                "same_predicted_genre": in_genre_pred == rec_genre_pred,
                "top_shared_features": top_aligned,
                "all_features_ranked": features,
            },
            indent=2,
        )
