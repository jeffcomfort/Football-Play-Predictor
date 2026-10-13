"""
Football Play Predictor
-----------------------
A Streamlit app for learning opponent tendencies from prior-game CSV/XLSX data
and ranking likely next offensive plays from a coach-entered situation.

Run:
    pip install -r requirements.txt
    streamlit run football_play_predictor.py

Important:
- One row should represent one offensive snap.
- Choose the column that identifies the play you want to predict (usually OFF PLAY).
- Only select features that are known BEFORE the snap. Do not use the actual
  result/play label or post-snap information as a predictor.
- Predictions are estimates based on the supplied data, not guarantees.
"""

from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

st.set_page_config(page_title="Football Play Predictor", page_icon="🏈", layout="wide")
st.title("🏈 Football Play Predictor")
st.caption("Import opponent scouting data, learn tendencies, and rank likely plays for a game situation.")

DEFAULT_FEATURES = [
    "OFF FORM", "WRS", "DOWN", "DISTANCE", "YARD LINE",
    "QUARTER", "TIME", "SCORE DIFFERENTIAL", "BACKFIELD",
    "PLAY STRENGTH", "FIELD ZONE"
]
LEAKAGE_HINTS = [
    "ACTUAL RESULT", "YARDS GAINED", "GAIN", "OUTCOME", "RESULT",
    "PLAY TYPE", "OFF PLAY", "PLAY DIRECTION", "PLAY CALL"
]

def read_uploaded_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        # Try common encodings and delimiter inference.
        raw = uploaded.getvalue()
        for enc in ("utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            except Exception:
                pass
        raise ValueError("Could not read this CSV. Save it as UTF-8 CSV or Excel.")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or Excel file.")

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normalized_col(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def guess_target(columns):
    for candidate in ("OFF PLAY", "PLAY", "PLAY CALL", "OFFENSIVE PLAY"):
        for c in columns:
            if normalized_col(c) == normalized_col(candidate):
                return c
    return columns[-1]

def guess_features(columns, target):
    selected = []
    for c in columns:
        n = normalized_col(c)
        if c == target:
            continue
        if any(normalized_col(h) == n for h in LEAKAGE_HINTS):
            continue
        if any(normalized_col(d) == n for d in DEFAULT_FEATURES):
            selected.append(c)
    return selected

def prepare_xy(df, target, features):
    work = df[features + [target]].copy()
    work = work.replace(r"^\s*$", np.nan, regex=True)
    work = work.dropna(subset=[target])
    # Convert values to strings so inconsistent Excel types don't break encoding.
    for c in features:
        work[c] = work[c].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        work[c] = work[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    work[target] = work[target].astype(str).str.strip()
    work = work[work[target] != ""]
    return work[features], work[target], work

def build_model(X, y, n_estimators, min_samples_leaf):
    categorical = list(X.columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ],
        remainder="drop"
    )
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

def mode_or_blank(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.mode().iloc[0] if not s.empty else ""

with st.sidebar:
    st.header("1. Upload scouting data")
    uploaded = st.file_uploader("Upload game data", type=["xlsx", "xls", "csv"])
    st.markdown("**Recommended:** one row per offensive snap, with consistent labels.")

if not uploaded:
    st.info("Upload an Excel or CSV scouting spreadsheet to begin.")
    st.markdown("""
    **Useful columns**
    - Pre-snap situation: `DOWN`, `DISTANCE`, `YARD LINE`, `QUARTER`, `TIME`
    - Formation/alignment: `OFF FORM`, `WRS`, `BACKFIELD`, `PLAY STRENGTH`
    - Prediction target: `OFF PLAY` (or your equivalent play-call column)
    - Optional context: opponent, game, score differential, field zone
    """)
    st.stop()

try:
    df = clean_column_names(read_uploaded_file(uploaded))
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The file needs at least two columns and some play rows.")
    st.stop()

st.subheader("2. Select what to predict")
left, right = st.columns([1, 2])
with left:
    target = st.selectbox("Play label / target column", df.columns.tolist(),
                          index=df.columns.tolist().index(guess_target(df.columns.tolist())))
    default_features = guess_features(df.columns.tolist(), target)
    candidates = [c for c in df.columns if c != target]
    if not default_features:
        default_features = candidates[:min(6, len(candidates))]
    features = st.multiselect(
        "Pre-snap predictor columns",
        options=candidates,
        default=default_features,
        help="Only choose information available before the snap. Avoid actual play/result columns."
    )
    n_estimators = st.slider("Model size", min_value=100, max_value=600, value=300, step=100)
    min_leaf = st.slider("Minimum examples per leaf", min_value=1, max_value=8, value=2)
with right:
    st.write("**Data preview**")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

if not features:
    st.warning("Select at least one pre-snap predictor column.")
    st.stop()
if target in features:
    st.error("The target column cannot also be a predictor.")
    st.stop()

X, y, clean = prepare_xy(df, target, features)
if len(clean) < 10:
    st.error(f"Only {len(clean)} usable rows. Add more snaps (at least 10; 100+ is a better starting point).")
    st.stop()
if y.nunique() < 2:
    st.error("The target column needs at least two different play labels.")
    st.stop()

st.subheader("3. Train and check the model")
class_counts = y.value_counts()
st.write(f"Training rows: **{len(y):,}** · Distinct play labels: **{y.nunique()}**")
rare = int((class_counts < 2).sum())
if rare:
    st.warning(f"{rare} play label(s) appear only once. Their probabilities will be unreliable.")

# Holdout evaluation when the dataset is large enough and every class can be stratified.
can_split = len(y) >= 30 and class_counts.min() >= 2
if can_split:
    test_size = max(0.2, y.nunique() / len(y))
    test_size = min(test_size, 0.35)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        eval_model = build_model(X_train, y_train, n_estimators, min_leaf)
        eval_model.fit(X_train, y_train)
        pred = eval_model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Holdout accuracy", f"{acc:.1%}")
        metric_col2.metric("Test snaps", f"{len(y_test):,}")
        metric_col3.metric("Most common play", str(class_counts.index[0]))
        st.caption("Holdout accuracy is the share of test snaps where the top-ranked play matched. It does not guarantee future-game accuracy.")
    except Exception as exc:
        st.warning(f"Could not create a reliable holdout score: {exc}")
else:
    st.info("Not enough examples per play label for a meaningful holdout test. Treat predictions as exploratory.")

model = build_model(X, y, n_estimators, min_leaf)
model.fit(X, y)

st.subheader("4. Enter the upcoming situation")
st.write("Fill in the pre-snap details for the current snap. Unknown fields can be left blank.")
with st.form("situation_form"):
    cols = st.columns(3)
    situation = {}
    for i, feature in enumerate(features):
        # Offer common observed values, plus a blank/unknown choice.
        values = clean[feature].dropna().astype(str).str.strip()
        values = values[values != ""]
        unique_values = values.value_counts().head(40).index.tolist()
        label = feature
        with cols[i % 3]:
            # Numeric-looking columns still use categories because many scouting labels are coded.
            if unique_values and len(values.unique()) <= 40:
                options = ["(Unknown / leave blank)"] + unique_values
                situation[feature] = st.selectbox(label, options, key=f"feature_{i}")
            else:
                situation[feature] = st.text_input(label, value="")
    top_n = st.slider("How many plays to show", 3, 10, 5)
    submitted = st.form_submit_button("Predict likely plays", type="primary")

if submitted:
    row = {}
    for feature in features:
        value = situation[feature]
        row[feature] = np.nan if value in ("", "(Unknown / leave blank)") else str(value).strip()
    input_df = pd.DataFrame([row], columns=features)
    probabilities = model.predict_proba(input_df)[0]
    labels = model.named_steps["model"].classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)[:top_n]
    results = pd.DataFrame(ranked, columns=["Predicted play", "Model probability"])
    results["Model probability"] = (results["Model probability"] * 100).round(1)
    results.insert(0, "Rank", range(1, len(results) + 1))
    st.subheader("Prediction ranking")
    st.dataframe(
        results.rename(columns={"Model probability": "Estimated probability (%)"}),
        use_container_width=True, hide_index=True
    )
    if ranked:
        st.markdown(f"**Top prediction:** {ranked[0][0]} — estimated model probability **{ranked[0][1]:.1%}**.")
    st.caption("Probabilities are model scores, not certainty. They may be poorly calibrated when the dataset is small, labels are inconsistent, or the upcoming opponent/game differs.")
    st.download_button(
        "Download prediction results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="football_play_predictions.csv",
        mime="text/csv"
    )

with st.expander("View tendency summaries"):
    st.markdown("**Most frequent plays in the uploaded sample**")
    freq = y.value_counts().head(20).rename_axis("Play").reset_index(name="Snaps")
    freq["Share (%)"] = (freq["Snaps"] / len(y) * 100).round(1)
    st.dataframe(freq, use_container_width=True, hide_index=True)

    st.markdown("**Play tendencies by selected situation fields**")
    possible_groups = [c for c in features if c in df.columns]
    group_by = st.selectbox("Group tendency by", ["(None)"] + possible_groups)
    if group_by != "(None)":
        tab = clean.groupby([group_by, target], dropna=False).size().reset_index(name="Snaps")
        tab["Share within group (%)"] = (
            tab["Snaps"] / tab.groupby(group_by)["Snaps"].transform("sum") * 100
        ).round(1)
        tab = tab.sort_values([group_by, "Snaps"], ascending=[True, False])
        st.dataframe(tab, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built for scouting support. Verify data labels and use coaching judgment; the tool does not account for every personnel, injury, weather, or in-game adjustment.")
