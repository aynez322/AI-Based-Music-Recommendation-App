import json
import os
import pickle
import re
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
    _load_fingerprints,
    _to_fingerprint_scale,
)

# clasificatorul xgboost foloseste 2 caracteristici suplimentare fata de cele 9 audio
CLASSIFIER_FEATURES = NUMERIC_FEATURES + ["mode", "key"]

_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
_MODEL_PATH = os.path.join(_DATA_DIR, "genre_classifier.pkl")

# cache-uri globale pentru model, encoder si explainer
_model: Optional[xgb.XGBClassifier] = None
_encoder: Optional[LabelEncoder] = None
_explainer: Optional[shap.TreeExplainer] = None


# mapeaza genuri detaliate de pe spotify la categorii largi pentru clasificator
def _consolidate_genre(genre: str) -> str:
    g = genre.lower().strip()

    # hip-hop si trap
    if re.search(r"\btrap\b|\bdrill\b", g):                  return "hip-hop"
    if re.search(r"hip.hop|rap\b|hip_hop", g):              return "hip-hop"
    if re.search(r"\bemo\b|post.hardcore|screamo", g):      return "rock"

    # metal inainte de rock pentru a evita suprapunerile
    if re.search(r"death.metal|black.metal|power.metal|thrash.metal|heavy.metal|"
                 r"doom.metal|gothic.metal|symphonic.metal|melodic.death|"
                 r"grindcore|metalcore|nu.metal|prog.*metal", g): return "metal"
    if re.search(r"\bmetal\b", g):                          return "metal"

    # punk si subgenuri care suna similar cu rock-ul
    if re.search(r"punk|post.punk|protopunk|new.wave|new_wave|hardcore", g): return "rock"

    # subgenuri de rock
    if re.search(r"grunge|prog.*rock|progressive.rock|classic.rock|hard.rock|"
                 r"blues.rock|rockabilly|garage.rock|post.rock|psychedelic.rock|"
                 r"brit.pop|britpop|beatlesque|brit.invasion|merseybeat|"
                 r"madchester|shoegaze|alt.rock", g):       return "rock"
    if re.search(r"\brock\b", g):                           return "rock"

    # r&b si soul
    if re.search(r"r&b|rnb|\bsoul\b|r-n-b|rhythm.and.blues|"
                 r"new.jack.swing|quiet.storm|souldies|motown", g): return "r&b"

    # subgenuri de pop inainte de electronic ca synthpop sa nu fie confundat
    if re.search(r"indie.pop|dream.pop|art.pop|power.pop|bubblegum|"
                 r"synth.pop|synthpop|electro.pop|electropop|"
                 r"eurodance|europop|teen.pop|dance.pop|"
                 r"chamber.pop|baroque.pop|twee.pop|jangle.pop", g): return "pop"

    # subgenuri electronice
    if re.search(r"\btechno\b", g):                         return "electronic"
    if re.search(r"\bhouse\b", g):                          return "electronic"
    if re.search(r"trance|hardstyle|dubstep|drum.and.bass|drum_and_bass|"
                 r"dnb|breakbeat|jungle|trip.hop|trip_hop|idm|"
                 r"hi.nrg|big.beat|club|garage\b|grime|"
                 r"future.bass|brostep|complextro|crunk", g): return "electronic"
    if re.search(r"edm|electronic|electro|dance\b|\bdance$", g): return "electronic"

    # jazz
    if re.search(r"\bjazz\b|bebop|\bswing\b|big.band|bop\b", g): return "jazz"

    # clasic si new age
    if re.search(r"\bclassical\b|orchestra|symphony|baroque|opera|"
                 r"chamber|new.age|new_age|piano\b|instrumental\b", g): return "classical"

    # blues
    if re.search(r"\bblues\b", g):                          return "blues"

    # funk si groove tratate ca r&b
    if re.search(r"\bfunk\b|groove\b", g):                  return "r&b"

    # disco tratat ca electronic
    if re.search(r"\bdisco\b", g):                          return "electronic"

    # ska
    if re.search(r"\bska\b", g):                            return "ska"

    # country si bluegrass
    if re.search(r"\bcountry\b|honky.tonk|bluegrass|americana|"
                 r"rockabilly|nashville|cowpunk", g):        return "country"

    # folk si singer-songwriter
    if re.search(r"\bfolk\b|singer.songwriter|acoustic\b|"
                 r"bossa|mpb|fado|flamenco|tango\b", g):    return "folk"

    # latin
    if re.search(r"latin|reggaeton|cumbia|salsa|bachata|dembow|"
                 r"samba|pagode|sertanejo|forro|axe|baile.funk|"
                 r"tango|bolero|mariachi|norteño", g):       return "latin"

    # reggae si dub
    if re.search(r"reggae|dancehall|\bdub\b|ska.punk", g):  return "reggae"

    # k-pop si pop asiatic
    if re.search(r"k.pop|kpop|j.pop|jpop|j.idol|j.dance|j.rock|"
                 r"cantopop|mandopop|j_pop", g):             return "k-pop"

    # afrobeats tratat ca world music
    if re.search(r"afrobeat|afro.pop|afropop|afroswing|afro\b", g): return "world"

    # ambient si lo-fi
    if re.search(r"ambient|lo.fi|lofi|chillwave|downtempo|"
                 r"chillout|chill.out|new.age|new_age|"
                 r"\bchill\b|sleep\b|study\b|meditation", g): return "ambient"

    # goth tratat ca metal
    if re.search(r"\bgoth\b|gothic", g):                    return "metal"

    # industrial tratat ca electronic
    if re.search(r"industrial", g):                         return "electronic"

    # muzica mondiala si etnica
    if re.search(r"world|tribal|ethnic|global|bollywood|filmi|"
                 r"bhangra|indian\b|malay|iranian|turkish|"
                 r"spanish\b|german\b|french\b|swedish\b|"
                 r"japanese\b|russian\b|polish\b|celtic|"
                 r"flamenco|fado|tango\b|soca|calypso|"
                 r"brazil\b|opm|enka|cumbia", g):            return "world"

    # indie si alternative tratate ca rock
    if re.search(r"indie|alternative", g):                   return "rock"

    # pop ca ultima optiune pentru orice nu a fost clasificat anterior
    if re.search(r"\bpop\b", g):                            return "pop"

    return genre


# incarca datele de antrenament pentru clasificatorul de gen
def _load_training_data() -> pd.DataFrame:
    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(
            "merged_dataset.parquet not found. Run scripts/build_final_dataset.py first."
        )
    df = pd.read_parquet(MERGED_PATH)
    df = df[df["track_genre"] != "unknown"].dropna(subset=NUMERIC_FEATURES + ["track_genre"])
    df = df.copy()
    for col in ["mode", "key"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    if len(df) < 1000:
        raise RuntimeError(
            "Not enough labelled songs to train the genre classifier "
            f"(found {len(df)}, need >= 1 000). "
            "Run scripts/build_final_dataset.py first."
        )
    df["track_genre"] = df["track_genre"].apply(_consolidate_genre)
    return df


# incarca modelul xgboost din cache sau il antreneaza daca nu exista
def _get_model_and_explainer() -> tuple:
    global _model, _encoder, _explainer
    if _model is not None:
        return _model, _encoder, _explainer

    if os.path.exists(_MODEL_PATH):
        print("Loading cached genre classifier...")
        with open(_MODEL_PATH, "rb") as fh:
            saved = pickle.load(fh)
        _model = saved["model"]
        _encoder = saved["encoder"]
    else:
        print("Training XGBoost genre classifier (first run — may take 1-2 min)...")
        df_train = _load_training_data()
        n_genres = df_train["track_genre"].nunique()
        print(f"  Training on {len(df_train):,} songs, {n_genres} consolidated genres")

        le = LabelEncoder()
        y = le.fit_transform(df_train["track_genre"])
        X = df_train[CLASSIFIER_FEATURES].values.astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        acc = model.score(X_te, y_te)
        print(f"  Genre classifier accuracy: {acc:.3f}")

        with open(_MODEL_PATH, "wb") as fh:
            pickle.dump({"model": model, "encoder": le}, fh)
        _model = model
        _encoder = le

    _explainer = shap.TreeExplainer(_model)
    return _model, _encoder, _explainer


# calculeaza valorile shap pentru clasa prezisa de clasificator
def _shap_for_class(
    explainer: shap.TreeExplainer, X_row: np.ndarray, pred_class: int
) -> np.ndarray:
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


# determina genul unui cantec din eticheta din dataset sau prin clasificatorul xgboost
def _resolve_genre(
    df: "pd.DataFrame",
    idx: int,
    model: xgb.XGBClassifier,
    le: LabelEncoder,
    X_row: np.ndarray,
) -> tuple[str, int]:
    if "track_genre" in df.columns:
        raw = str(df.loc[idx, "track_genre"])
        consolidated = _consolidate_genre(raw)
        if consolidated in le.classes_:
            class_idx = int(le.transform([consolidated])[0])
            return consolidated, class_idx

    class_idx = int(model.predict(X_row)[0])
    return le.inverse_transform([class_idx])[0], class_idx


# tool care explica similaritatea dintre doua cantece folosind shap si xgboost
class SHAPExplainInput(BaseModel):
    input_song: str = Field(..., description="Title of the input/reference song")
    recommended_song: str = Field(..., description="Title of the recommended song")
    input_artist: str = Field(default="", description="Artist of the input song (optional)")
    recommended_artist: str = Field(default="", description="Artist of the recommended song (optional)")


class SHAPExplainerTool(BaseTool):
    name: str = "shap_explain_similarity"
    description: str = (
        "Explains why a recommended song is similar to the input song using a trained "
        "XGBoost genre classifier and SHAP (SHapley Additive exPlanations). "
        "Reveals which audio features (energy, valence, tempo, acousticness, etc.) are "
        "the main drivers of similarity and enriches the explanation with genre "
        "fingerprint key signals."
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
            fps = _load_fingerprints()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        in_idx = _find_index(df, input_song, input_artist)
        rec_idx = _find_index(df, recommended_song, recommended_artist)

        if in_idx is None:
            return json.dumps({"error": f"'{input_song}' not found in dataset."})
        if rec_idx is None:
            return json.dumps({"error": f"'{recommended_song}' not found in dataset."})

        # construieste vectorii de caracteristici pentru clasificator; coloanele absente devin nan
        def _row_to_classifier_vec(idx):
            return np.array([
                float(df.loc[idx, f]) if f in df.columns else np.nan
                for f in CLASSIFIER_FEATURES
            ], dtype=float).reshape(1, -1)

        in_X  = _row_to_classifier_vec(in_idx)
        rec_X = _row_to_classifier_vec(rec_idx)
        in_X  = np.where(in_X  == -1, np.nan, in_X)
        rec_X = np.where(rec_X == -1, np.nan, rec_X)

        in_genre_pred, in_pred   = _resolve_genre(df, in_idx,  model, le, in_X)
        rec_genre_pred, rec_pred = _resolve_genre(df, rec_idx, model, le, rec_X)

        in_shap = _shap_for_class(explainer, in_X, in_pred)
        rec_shap = _shap_for_class(explainer, rec_X, rec_pred)

        in_vals = {
            f: round(float(df.loc[in_idx, f]), 4) if f in df.columns else None
            for f in CLASSIFIER_FEATURES
        }
        rec_vals = {
            f: round(float(df.loc[rec_idx, f]), 4) if f in df.columns else None
            for f in CLASSIFIER_FEATURES
        }

        features = []
        for i, feat in enumerate(CLASSIFIER_FEATURES):
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

        # imbogateste explicatia cu semnalele cheie ale genului comun din amprente
        in_fp_key = in_genre_pred.lower()
        rec_fp_key = rec_genre_pred.lower()
        fp_key_signals_input = fps.get(in_fp_key, {}).get("key_signals", "")
        fp_key_signals_rec = fps.get(rec_fp_key, {}).get("key_signals", "")

        # calculeaza similaritatea cosinus intre amprente pentru cele doua cantece
        in_vec = _to_fingerprint_scale(df.loc[in_idx])
        rec_vec = _to_fingerprint_scale(df.loc[rec_idx])
        in_norm = np.linalg.norm(in_vec) or 1.0
        rec_norm = np.linalg.norm(rec_vec) or 1.0
        fp_cosine = round(float(np.dot(in_vec, rec_vec) / (in_norm * rec_norm)), 4)

        return json.dumps(
            {
                "input_song": str(df.loc[in_idx, "track_name"]),
                "recommended_song": str(df.loc[rec_idx, "track_name"]),
                "input_predicted_genre": in_genre_pred,
                "recommended_predicted_genre": rec_genre_pred,
                "same_predicted_genre": in_genre_pred == rec_genre_pred,
                "fingerprint_cosine_similarity": fp_cosine,
                "input_fingerprint_key_signals": fp_key_signals_input,
                "recommended_fingerprint_key_signals": fp_key_signals_rec,
                "top_shared_features": top_aligned,
                "all_features_ranked": features,
            },
            indent=2,
        )
