import streamlit as st
import joblib
import pandas as pd
import os
import json
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="Orbitly",
    page_icon="🪐",
    layout="wide"
)

# ==========================================================
# GLOBAL STYLING
# ==========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #dbeafe 0%, #c7d2fe 45%, #bae6fd 100%);
        font-family: 'Inter', sans-serif;
    }

    h1 {
        font-family: 'Poppins', sans-serif !important;
        color: #1e3a5f !important;
        font-weight: 800 !important;
        font-size: 3rem !important;
    }
    h2 {
        font-family: 'Poppins', sans-serif !important;
        color: #1e3a5f !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #1e3a5f !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    p, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: #1e3a5f !important;
        font-size: 1.05rem !important;
    }

    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        font-family: 'Inter', sans-serif !important;
        color: #1e3a5f !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #1e3a5f !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #0f2a47 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border-radius: 22px !important;
        box-shadow: 0 10px 30px rgba(30, 58, 95, 0.15);
        border: 1px solid rgba(255,255,255,0.7) !important;
        padding: 10px;
    }

    div.stButton > button {
        font-family: 'Poppins', sans-serif !important;
        background: linear-gradient(135deg, #4f86c6, #6ea8e0);
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 0.75em 1.4em;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: 0.2s ease;
        box-shadow: 0 4px 14px rgba(79, 134, 198, 0.4);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #3b6ba5, #4f86c6);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 134, 198, 0.55);
    }

    .blob {
        position: fixed;
        border-radius: 50%;
        filter: blur(100px);
        z-index: -1;
        opacity: 0.45;
        pointer-events: none;
    }
    .blob-mint   { width: 320px; height: 320px; background: #95f5a8; top: -5%;  left: -5%; }
    .blob-peach  { width: 300px; height: 300px; background: #ffc27a; bottom: -8%; right: -5%; }
    .blob-lav    { width: 280px; height: 280px; background: #d79bfa; top: 40%; left: 95%; }
    .blob-sky    { width: 260px; height: 260px; background: #5fd4ea; bottom: 10%; left: -8%; }

    @keyframes floatSlow {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    .blob { animation: floatSlow 10s ease-in-out infinite; }

    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .badge-high { background: #ffb3b3; color: #7a0c12; }
    .badge-mod  { background: #ffe29a; color: #7a5b0c; }
    .badge-low  { background: #a9f5b4; color: #1d5c2a; }
    .badge-seg  { background: #aecbfa; color: #15325c; }
    </style>

    <div class="blob blob-mint"></div>
    <div class="blob blob-peach"></div>
    <div class="blob blob-lav"></div>
    <div class="blob blob-sky"></div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# SETTINGS (change these here instead of hunting through the code)
# ==========================================================
MODEL_NAME = "openai/gpt-oss-20b"
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_TOKENS = 1500          # reasoning models spend part of this "thinking"
MAX_LLM_CALLS_PER_SESSION = 20  # basic guard so one visitor can't burn your Groq quota
RISK_HIGH = 0.6
RISK_MODERATE = 0.3
BATCH_MAX_ROWS = 10000

# Approved retention offers. The AI may only choose from this menu (edit freely).
OFFER_MENU = [
    "Discount for moving to a 1-year or 2-year contract",
    "Free Tech Support or Online Security for 3 months",
    "Small bill credit for switching to auto-pay",
    "Loyalty bill credit for long-standing customers",
    "Plan right-sizing (move to a cheaper plan that still fits their usage)",
    "Free streaming add-on trial",
]

# --- Load saved artifacts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    def p(name):
        return os.path.join(BASE_DIR, name)
    log_reg = joblib.load(p('log_reg_model.pkl'))
    scaler = joblib.load(p('scaler.pkl'))
    kmeans = joblib.load(p('kmeans_model.pkl'))
    scaler_cluster = joblib.load(p('scaler_cluster.pkl'))
    model_columns = joblib.load(p('model_columns.pkl'))
    return log_reg, scaler, kmeans, scaler_cluster, model_columns

log_reg, scaler, kmeans, scaler_cluster, model_columns = load_models()


def load_json_file(name):
    """Optional files produced in Colab. Returns None if missing or unreadable."""
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Used only if segment_profiles.json (made in Colab) is not found.
DEFAULT_SEGMENT_PROFILES = {
    0: {"name": "Phone-Only (No Internet)",
        "description": "Mid-tenure customers with phone service but no internet, paying very low monthly bills (about $21).",
        "churn_rate": 0.07},
    1: {"name": "Internet-Only (No Phone)",
        "description": "Mid-tenure customers on DSL internet with no phone line, paying moderate monthly bills (about $42).",
        "churn_rate": 0.25},
    2: {"name": "New High-Bill Customers",
        "description": "Newer customers (short tenure) paying high monthly bills, often fiber optic, usually month-to-month contracts.",
        "churn_rate": 0.47},
    3: {"name": "Loyal High-Value Customers",
        "description": "Very long-tenure, high-paying, loyal customers.",
        "churn_rate": 0.15},
}

_loaded_profiles = load_json_file("segment_profiles.json")
SEGMENT_PROFILES = (
    {int(k): v for k, v in _loaded_profiles.items()} if _loaded_profiles else DEFAULT_SEGMENT_PROFILES
)
MODEL_METRICS = load_json_file("model_metrics.json")


def get_profile(segment):
    return SEGMENT_PROFILES.get(
        segment,
        {"name": f"Segment {segment}", "description": "No profile available.", "churn_rate": None},
    )


# ==========================================================
# FEATURE ENCODING (one place, one set of known labels)
# ==========================================================
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

KNOWN_LABELS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
REQUIRED_BATCH_COLUMNS = NUMERIC_COLUMNS + list(KNOWN_LABELS.keys())


def check_model_columns(columns):
    """Compare the model's columns with the labels this app uses.
    A mismatch would silently give wrong predictions, so we surface it."""
    problems = []
    for col in NUMERIC_COLUMNS:
        if col not in columns:
            problems.append(f"missing numeric column '{col}'")
    for col in columns:
        if col in NUMERIC_COLUMNS:
            continue
        matched = False
        for field, labels in KNOWN_LABELS.items():
            prefix = field + "_"
            if col.startswith(prefix):
                matched = True
                if col[len(prefix):] not in labels:
                    problems.append(f"column '{col}' has a label this app doesn't use")
        if not matched:
            problems.append(f"unrecognised column '{col}'")
    return problems


_column_problems = check_model_columns(list(model_columns))
if _column_problems:
    st.warning(
        "⚠️ The app and the saved model don't fully agree on feature names, so predictions may be wrong: "
        + "; ".join(_column_problems[:5])
        + (" ..." if len(_column_problems) > 5 else "")
    )


def build_features(d):
    """One customer (dict) -> one-row DataFrame in the exact column order the model expects."""
    row = pd.DataFrame(0.0, index=[0], columns=model_columns)
    for col in NUMERIC_COLUMNS:
        if col in row.columns:
            row.at[0, col] = float(d[col])
    for field in KNOWN_LABELS:
        col_name = f"{field}_{d[field]}"
        if col_name in row.columns:  # reference categories were dropped in training
            row.at[0, col_name] = 1.0
    return row


def predict_customer(d):
    X = build_features(d)
    X_scaled = scaler.transform(X)
    proba = float(log_reg.predict_proba(X_scaled)[0, 1])
    segment = int(kmeans.predict(scaler_cluster.transform(X))[0])
    return {"proba": proba, "segment": segment, "scaled": X_scaled}


def risk_level_for(proba):
    if proba >= RISK_HIGH:
        return "High"
    if proba >= RISK_MODERATE:
        return "Moderate"
    return "Low"


def friendly_feature_name(col):
    special = {
        "tenure": "Tenure (months)",
        "MonthlyCharges": "Monthly bill",
        "TotalCharges": "Total charges so far",
        "SeniorCitizen": "Senior citizen",
    }
    if col in special:
        return special[col]
    if "_" in col:
        field, value = col.split("_", 1)
        return f"{field}: {value}"
    return col


def top_drivers(X_scaled, n=3):
    """Which features push this customer's risk up/down compared with an average customer.
    contribution = model coefficient x the customer's scaled value."""
    try:
        contrib = log_reg.coef_[0] * X_scaled[0]
        df = pd.DataFrame({"feature": list(model_columns), "contribution": contrib})
        up = df[df.contribution > 0].sort_values("contribution", ascending=False).head(n)
        down = df[df.contribution < 0].sort_values("contribution").head(n)
        return (
            [friendly_feature_name(f) for f in up.feature],
            [friendly_feature_name(f) for f in down.feature],
        )
    except Exception:
        return None, None


def input_warnings(monthly_charges, internet_service, phone_service="Yes"):
    """Flag inputs far from what the model saw in training."""
    msgs = []
    bill_usd = monthly_charges
    if internet_service == "No" and bill_usd > 30:
        msgs.append(
            "No-internet customers in the training data pay only about $18–26 a month. "
            "A bill this high is outside what the model has seen, so the result may be unreliable."
        )
    if internet_service == "Fiber optic" and bill_usd < 50:
        msgs.append(
            "Fiber customers in the training data usually pay much more than this, "
            "so the result may be less reliable."
        )
    if phone_service == "No" and internet_service == "No":
        msgs.append(
            "The training data has no customers with neither phone nor internet, "
            "so this combination can't be scored reliably."
        )
    if phone_service == "No" and internet_service == "Fiber optic":
        msgs.append(
            "In the training data, customers without phone service all use DSL internet, "
            "so this combination is outside what the model has seen."
        )
    return msgs


# ==========================================================
# LLM HELPER
# ==========================================================
def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return key


def ask_llm(client, prompt):
    """Returns (text, error_message). Never raises."""
    calls = st.session_state.get("llm_calls", 0)
    if calls >= MAX_LLM_CALLS_PER_SESSION:
        return None, "AI text limit reached for this session. Refresh the page to start a new session."
    st.session_state.llm_calls = calls + 1  # count attempts, so failures can't loop forever
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return None, "The AI returned an empty answer. Please try again."
        return text, None
    except Exception:
        return None, "Couldn't reach the AI assistant right now. Please try again in a moment."


# ==========================================================
# BATCH SCORING (CSV upload)
# ==========================================================
TEMPLATE_ROW = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 1026.0,
}


def score_batch(raw):
    """Returns (result_df or None, notes list, skipped_row_count)."""
    raw = raw.reset_index(drop=True)
    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in raw.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"], 0

    df = raw.head(BATCH_MAX_ROWS).copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0})
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # blank TotalCharges (brand-new customers) -> estimate as tenure x monthly
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    valid = df[NUMERIC_COLUMNS].notna().all(axis=1)
    notes = []
    for field, labels in KNOWN_LABELS.items():
        bad = ~df[field].isin(labels)
        if bad.any():
            notes.append(
                f"{int(bad.sum())} row(s) have an unrecognised value in '{field}' "
                f"(allowed: {', '.join(labels)})."
            )
        valid &= ~bad
    skipped = int((~valid).sum())
    good = df[valid]
    if good.empty:
        return None, notes, skipped

    X = pd.DataFrame(0.0, index=good.index, columns=model_columns)
    for col in NUMERIC_COLUMNS:
        if col in X.columns:
            X[col] = good[col].astype(float)
    for field, labels in KNOWN_LABELS.items():
        for label in labels:
            col_name = f"{field}_{label}"
            if col_name in X.columns:
                X[col_name] = (good[field] == label).astype(float)

    proba = log_reg.predict_proba(scaler.transform(X))[:, 1]
    segments = kmeans.predict(scaler_cluster.transform(X))

    out = raw.loc[good.index].copy()
    out["churn_risk_score"] = proba.round(3)
    out["risk_level"] = pd.cut(
        proba, bins=[-0.01, RISK_MODERATE, RISK_HIGH, 1.01],
        labels=["Low", "Moderate", "High"], right=False
    )
    out["segment"] = [get_profile(int(s))["name"] for s in segments]
    out = out.sort_values("churn_risk_score", ascending=False)
    return out, notes, skipped


# ==========================================================
# BRING YOUR OWN DATA (train a quick model on any churn CSV)
# ==========================================================
CUSTOM_MAX_ROWS = 50000
CUSTOM_MIN_ROWS = 200
CUSTOM_MIN_PER_CLASS = 30       # need at least this many churners AND non-churners
LEAKAGE_AUC = 0.95              # a single column this predictive is suspicious
CUSTOM_PREVIEW_ROWS = 200


def _is_text(series):
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series))


def coerce_numeric_like(df):
    """Text columns that are really numbers (e.g. blanks mixed with digits) -> numeric."""
    df = df.copy()
    for col in df.columns:
        if _is_text(df[col]):
            non_null = int(df[col].notna().sum())
            if non_null == 0:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if int(converted.notna().sum()) >= 0.95 * non_null:
                df[col] = converted
    return df


def guess_target_index(columns):
    for i, c in enumerate(columns):
        if any(k in str(c).lower() for k in ("churn", "exited", "attrition", "left", "cancel")):
            return i
    return 0


def guess_positive_index(values):
    likely = {"1", "yes", "true", "y", "churn", "churned", "left", "exited"}
    for i, v in enumerate(values):
        if str(v).strip().lower() in likely:
            return i
    return len(values) - 1


def auto_ignore(df, target):
    """Columns that can't help predict churn: constants, ID-like columns, free text."""
    n = len(df)
    out = []
    for col in df.columns:
        if col == target:
            continue
        nun = df[col].nunique(dropna=True)
        id_like_type = _is_text(df[col]) or pd.api.types.is_integer_dtype(df[col])
        if nun <= 1:
            out.append(col)
        elif id_like_type and nun == n:
            out.append(col)
        elif _is_text(df[col]) and nun > 50:
            out.append(col)
    return out


def _make_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def _single_column_auc(X, y, col, is_numeric, cv):
    try:
        pipe = _make_pipeline([col] if is_numeric else [], [] if is_numeric else [col])
        p = cross_val_predict(pipe, X[[col]], y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def find_leaky_columns(X, y, num_cols, cat_cols):
    """Columns that predict churn almost perfectly on their own (often recorded after the customer left)."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    flagged = []
    for col in (num_cols + cat_cols)[:40]:
        auc = _single_column_auc(X, y, col, col in num_cols, cv)
        if auc is not None and auc >= LEAKAGE_AUC:
            flagged.append((col, auc))
    return sorted(flagged, key=lambda t: -t[1])


def train_custom_model(df, target, positive_value, ignore_cols):
    """Returns a dict of metrics, drivers and scored customers. Raises ValueError with a friendly message."""
    data = df.dropna(subset=[target]).reset_index(drop=True)
    y = (data[target].astype(str) == str(positive_value)).astype(int)
    if len(y) < CUSTOM_MIN_ROWS:
        raise ValueError(f"Only {len(y)} usable rows. At least {CUSTOM_MIN_ROWS} are needed for a trustworthy model.")
    churners = int(y.sum())
    if min(churners, len(y) - churners) < CUSTOM_MIN_PER_CLASS:
        raise ValueError(
            f"Only {min(churners, len(y) - churners)} rows in the smaller group. "
            f"At least {CUSTOM_MIN_PER_CLASS} churners and {CUSTOM_MIN_PER_CLASS} non-churners are needed."
        )

    X = data.drop(columns=[target] + [c for c in ignore_cols if c in data.columns])
    X = coerce_numeric_like(X)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable columns are left after ignoring the selected ones.")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].fillna("Missing").astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = _make_pipeline(num_cols, cat_cols)
    # every customer is scored by a model that never saw that customer
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    base_rate = float(y.mean())
    top_n = max(1, int(round(len(y) * 0.10)))
    top_idx = pd.Series(proba).sort_values(ascending=False).index[:top_n]
    top_rate = float(y.iloc[top_idx].mean())
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "churn_rate": base_rate,
        "baseline_accuracy": max(base_rate, 1 - base_rate),
        "top10_churn_rate": top_rate,
        "top10_lift": top_rate / base_rate if base_rate > 0 else 0.0,
        "rows": int(len(y)),
    }

    # which columns matter (fit once on everything)
    final = _make_pipeline(num_cols, cat_cols).fit(X, y)
    names = list(final.named_steps["pre"].get_feature_names_out())
    coefs = final.named_steps["clf"].coef_[0]
    clean = []
    for nm in names:
        nm = nm.replace("num__", "", 1).replace("cat__", "", 1)
        for c in cat_cols:
            if nm.startswith(c + "_"):
                nm = f"{c} = {nm[len(c) + 1:]}"
                break
        clean.append(nm)
    drv = pd.DataFrame({"feature": clean, "coef": coefs})
    drivers_up = drv.sort_values("coef", ascending=False).head(5)
    drivers_up = drivers_up[drivers_up.coef > 0].feature.tolist()
    drivers_down = drv.sort_values("coef").head(5)
    drivers_down = drivers_down[drivers_down.coef < 0].feature.tolist()

    out = data.copy()
    out["churn_risk_score"] = pd.Series(proba).round(3)
    pct = out["churn_risk_score"].rank(ascending=False, pct=True)
    out["risk_tier"] = pd.cut(pct, bins=[0, 0.10, 0.30, 1.0],
                              labels=["High", "Moderate", "Low"], include_lowest=True)
    out = out.sort_values("churn_risk_score", ascending=False)

    return {
        "metrics": metrics,
        "drivers_up": drivers_up,
        "drivers_down": drivers_down,
        "leaky": find_leaky_columns(X, y, num_cols, cat_cols),
        "scored": out,
        "n_features": int(X.shape[1]),
    }


if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ==========================================================
# SCREEN 1: WELCOME
# ==========================================================
if st.session_state.page == "welcome":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <svg width="100%" height="180" viewBox="0 0 400 180">
                <circle cx="200" cy="90" r="80" fill="none" stroke="#4f86c6" stroke-width="2" opacity="0.5"/>
                <circle cx="200" cy="90" r="58" fill="none" stroke="#8ecae6" stroke-width="2" opacity="0.6"/>
                <circle cx="280" cy="90" r="7" fill="#4f86c6"/>
                <circle cx="145" cy="40" r="5" fill="#ffb703"/>
                <circle cx="200" cy="90" r="18" fill="#1e3a5f"/>
            </svg>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<h1 style='text-align: center; margin-top:-20px;'>🪐 Orbitly</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 1.3rem !important; font-weight:600;'>"
            "AI-Powered Customer Churn Prediction & Retention Assistant</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>"
            "This tool predicts whether a telecom customer is likely to churn, "
            "identifies their behavioral segment, and generates a personalized "
            "retention offer for high-risk customers — powered by Logistic Regression, "
            "K-Means clustering, and an LLM (via Groq)."
            "</p>",
            unsafe_allow_html=True
        )
        st.caption(
            "Built on the public Telco churn dataset (priced in US dollars), "
            "so treat results as a demonstration."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
        if st.button("📁 Score a customer file (CSV)", use_container_width=True):
            st.session_state.page = "batch"
            st.rerun()
        if st.button("🧪 Bring your own data", use_container_width=True):
            st.session_state.page = "custom"
            st.rerun()

# ==========================================================
# SCREEN 2: FORM
# ==========================================================
elif st.session_state.page == "form":
    with st.container(border=True):
        st.title("📋 Enter Customer Details")
        st.write("Fill in the customer's profile to predict churn risk.")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Account Details")
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 36.0, step=1.0)

            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            INTERNET_LABELS = {
                "WiFi / Broadband (DSL)": "DSL",
                "Fiber Cable (High-Speed)": "Fiber optic",
                "No Internet Connection": "No"
            }
            internet_service_label = st.selectbox("Internet Service", list(INTERNET_LABELS.keys()))
            internet_service = INTERNET_LABELS[internet_service_label]
            PAYMENT_LABELS = {
                "UPI / Net Banking (pay manually each month)": "Electronic check",
                "Cheque / Mailed Payment": "Mailed check",
                "Bank Auto-Debit (standing instruction)": "Bank transfer (automatic)",
                "Credit/Debit Card (auto-pay)": "Credit card (automatic)"
            }
            payment_method_label = st.selectbox("Payment Method", list(PAYMENT_LABELS.keys()))
            payment_method = PAYMENT_LABELS[payment_method_label]
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

        with col2:
            st.subheader("Customer Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

            st.subheader("Add-on Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes"], disabled=(phone_service == "No"))
            online_security = st.selectbox("Online Security", ["No", "Yes"], disabled=(internet_service == "No"))
            online_backup = st.selectbox("Online Backup", ["No", "Yes"], disabled=(internet_service == "No"))
            device_protection = st.selectbox("Device Protection", ["No", "Yes"], disabled=(internet_service == "No"))
            tech_support = st.selectbox("Tech Support", ["No", "Yes"], disabled=(internet_service == "No"))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], disabled=(internet_service == "No"))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], disabled=(internet_service == "No"))

        for msg in input_warnings(monthly_charges, internet_service, phone_service):
            st.warning(msg)

        st.markdown("---")
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            if st.button("⬅ Back", use_container_width=True):
                st.session_state.page = "welcome"
                st.rerun()
        with nav_col2:
            if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):
                # Handle 3-state categories exactly matching Telco training dataset
                def format_addon(val):
                    return "No internet service" if internet_service == "No" else val

                st.session_state.inputs = {
                    'gender': gender,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': "No phone service" if phone_service == "No" else multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': format_addon(online_security),
                    'OnlineBackup': format_addon(online_backup),
                    'DeviceProtection': format_addon(device_protection),
                    'TechSupport': format_addon(tech_support),
                    'StreamingTV': format_addon(streaming_tv),
                    'StreamingMovies': format_addon(streaming_movies),
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': float(tenure * monthly_charges)
                }
                st.session_state.page = "results"
                st.rerun()

# ==========================================================
# SCREEN 3: RESULTS
# ==========================================================
elif st.session_state.page == "results":
    with st.container(border=True):
        st.title("📈 Prediction Results")
        st.markdown("---")

        input_dict = st.session_state.inputs
        result = predict_customer(input_dict)
        churn_proba = result["proba"]
        segment = result["segment"]
        profile = get_profile(segment)
        risk_level = risk_level_for(churn_proba)

        for msg in input_warnings(input_dict['MonthlyCharges'], input_dict['InternetService'], input_dict['PhoneService']):
            st.warning(msg)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Churn Risk Score", f"{churn_proba:.0%}")
            if risk_level == "High":
                st.markdown('<span class="badge badge-high">⚠️ High Risk</span>', unsafe_allow_html=True)
            elif risk_level == "Moderate":
                st.markdown('<span class="badge badge-mod">🟠 Moderate Risk</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-low">✅ Low Risk</span>', unsafe_allow_html=True)
            st.progress(float(churn_proba))
            st.caption("A relative score from the model, not an exact probability.")
        with res_col2:
            st.metric("Customer Segment", profile["name"])
            st.markdown(f'<span class="badge badge-seg">🧩 Segment {segment}</span>', unsafe_allow_html=True)
            if profile.get("churn_rate") is not None:
                st.caption(
                    f"Customers in this segment churned about {profile['churn_rate']:.0%} "
                    "of the time in the training data."
                )

        # ---------- Why this score? ----------
        st.markdown("---")
        st.subheader("🔍 Why this score?")
        drivers_up, drivers_down = top_drivers(result["scaled"])
        if drivers_up is None:
            st.caption("Couldn't work out the main factors for this customer.")
        else:
            why_col1, why_col2 = st.columns(2)
            with why_col1:
                st.markdown("**Pushing risk up**")
                if drivers_up:
                    for name in drivers_up:
                        st.write(f"🔺 {name}")
                else:
                    st.write("Nothing notable")
            with why_col2:
                st.markdown("**Pulling risk down**")
                if drivers_down:
                    for name in drivers_down:
                        st.write(f"🔻 {name}")
                else:
                    st.write("Nothing notable")
            st.caption("Compared with an average customer in the training data.")

        # ---------- AI explanations ----------
        st.markdown("---")
        st.subheader("🧩 What does this segment mean?")
        api_key = get_groq_api_key()

        if not api_key:
            st.error(
                "GROQ_API_KEY not found. Make sure you have a `.env` file in this folder "
                "containing: GROQ_API_KEY=your_key_here"
            )
        else:
            client = Groq(api_key=api_key)

            # reset saved LLM text when the customer's inputs change
            cache_key = str(sorted(input_dict.items()))
            if st.session_state.get("llm_key") != cache_key:
                st.session_state.llm_key = cache_key
                st.session_state.llm_segment = None
                st.session_state.llm_offer = None

            churn_text = (
                f"{profile['churn_rate']:.0%}" if profile.get("churn_rate") is not None else "unknown"
            )
            segment_prompt = f"""
A telecom customer has been placed into the segment "{profile['name']}" by a clustering model,
based on overall similarity across many features together (not any single trait alone).

General pattern typically seen in this segment (from real data analysis):
"{profile['description']}" Historical churn rate of the segment: {churn_text}.

This specific customer's actual details:
- Tenure: {input_dict['tenure']} months
- Monthly charges: ${input_dict['MonthlyCharges']:.2f}
- Contract type: {input_dict['Contract']}
- Internet service: {input_dict['InternetService']}

In 2-3 simple, non-technical sentences, explain to a business client what this segment
generally represents, and briefly note if/how this specific customer differs from the
segment's typical pattern (e.g. if their contract type differs from what's typical).
Avoid jargon like "K-Means" or "clustering" — speak in plain business terms.
"""
            if st.session_state.llm_segment is None:
                with st.spinner("Generating segment insight..."):
                    text, err = ask_llm(client, segment_prompt)
                if err:
                    st.warning(err)
                else:
                    st.session_state.llm_segment = text

            if st.session_state.llm_segment:
                st.info(st.session_state.llm_segment)

            st.markdown("---")

            if risk_level in ("High", "Moderate"):
                st.warning(f"⚠️ **{risk_level} churn risk detected**")
                drivers_text = ", ".join(drivers_up) if drivers_up else "no single strong factor"
                menu_text = "\n".join(f"- {o}" for o in OFFER_MENU)
                offer_prompt = f"""
A telecom customer has: tenure {input_dict['tenure']} months, monthly charges ${input_dict['MonthlyCharges']:.2f},
contract type {input_dict['Contract']}, internet service {input_dict['InternetService']},
churn risk score {churn_proba:.0%}, segment "{profile['name']}".
Main factors pushing their risk up (compared with an average customer): {drivers_text}.

Choose the 1-2 most suitable retention offers from this approved menu ONLY:
{menu_text}

Rules:
- Do not invent offers that are not on the menu.
- Do not state specific discount amounts or percentages; say the business team will set the amount.
- In 2-3 simple sentences, say which offer(s) you chose and why, linking them to the risk factors above.
"""
                if st.session_state.llm_offer is None:
                    with st.spinner("Contacting retention assistant..."):
                        text, err = ask_llm(client, offer_prompt)
                    if err:
                        st.warning(err)
                    else:
                        st.session_state.llm_offer = text

                if st.session_state.llm_offer:
                    st.success(f"💡 **Suggested Retention Offer:**\n\n{st.session_state.llm_offer}")
            else:
                st.info("✅ **Low churn risk** — no intervention needed right now.")

        # ---------- Notes & model info ----------
        st.markdown("---")
        st.caption(
            "This demo uses a model trained on the public Telco churn dataset, which is priced in US dollars. "
            "Total charges are estimated as tenure × monthly bill. "
            "Treat results as a demonstration, not business advice."
        )
        if MODEL_METRICS:
            with st.expander("ℹ️ About the model"):
                metric_labels = [
                    ("accuracy", "Accuracy"), ("precision", "Precision (churners)"),
                    ("recall", "Recall (churners)"), ("roc_auc", "ROC-AUC"),
                ]
                shown = [(k, label) for k, label in metric_labels if k in MODEL_METRICS]
                if shown:
                    cols = st.columns(len(shown))
                    for col, (k, label) in zip(cols, shown):
                        col.metric(label, f"{MODEL_METRICS[k]:.2f}")
                st.caption(
                    "Measured on held-out test data. Only about 1 in 4 customers in this dataset churns, "
                    "so recall and ROC-AUC say more than accuracy alone."
                )

        st.markdown("---")
        if st.button("🔄 Try Another Customer", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

# ==========================================================
# SCREEN 4: BATCH SCORING
# ==========================================================
elif st.session_state.page == "batch":
    with st.container(border=True):
        st.title("📁 Score a Customer File")
        st.write(
            "Upload a CSV in the same format as the Telco churn dataset (charges in USD). "
            f"Up to {BATCH_MAX_ROWS:,} rows are scored."
        )
        template_csv = pd.DataFrame([TEMPLATE_ROW]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV template", template_csv,
                           file_name="orbitly_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception:
                raw = None
                st.error("Couldn't read that file as a CSV.")

            if raw is not None:
                if len(raw) > BATCH_MAX_ROWS:
                    st.info(f"File has {len(raw):,} rows; only the first {BATCH_MAX_ROWS:,} are scored.")
                scored, notes, skipped = score_batch(raw)
                missing_cols = any(n.startswith("Missing columns") for n in notes)
                if missing_cols:
                    st.error(
                        "This file doesn't look like telecom customer data, so the telecom model can't score it. "
                        "Use the template above for the expected columns, or choose **Bring your own data** on the "
                        "home page to train a model on your own churn file."
                    )
                    with st.expander("See which columns are missing"):
                        st.write(notes[0])
                else:
                    for note in notes:
                        st.warning(note)
                if scored is None:
                    if not missing_cols:
                        st.error("No rows could be scored. Check the template for the expected columns and values.")
                else:
                    if skipped:
                        st.warning(f"{skipped} row(s) were skipped because of missing or unrecognised values.")
                    counts = scored["risk_level"].value_counts()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("High risk", int(counts.get("High", 0)))
                    m2.metric("Moderate risk", int(counts.get("Moderate", 0)))
                    m3.metric("Low risk", int(counts.get("Low", 0)))
                    st.dataframe(scored.head(200), use_container_width=True)
                    st.caption(f"Showing the 200 highest-risk rows. The download has all {len(scored):,} scored rows.")
                    st.download_button(
                        "⬇ Download scored file",
                        scored.to_csv(index=False).encode("utf-8"),
                        file_name="orbitly_scored.csv", mime="text/csv",
                    )
                    st.caption("Sorted from highest to lowest risk. Same model and caveats as the single-customer view.")

        st.markdown("---")
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()

# ==========================================================
# SCREEN 5: BRING YOUR OWN DATA
# ==========================================================
elif st.session_state.page == "custom":
    with st.container(border=True):
        st.title("🧪 Bring Your Own Data")
        st.write(
            "Upload any customer CSV that has a churn column. Orbitly trains a quick model on your file, "
            "checks how well it works, and scores every customer. This is separate from the telecom model."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="custom_upload")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception:
                raw = None
                st.error("Couldn't read that file as a CSV.")

            if raw is not None:
                if len(raw) > CUSTOM_MAX_ROWS:
                    st.info(f"File has {len(raw):,} rows; only the first {CUSTOM_MAX_ROWS:,} are used.")
                raw = raw.head(CUSTOM_MAX_ROWS).reset_index(drop=True)
                cols = list(raw.columns)

                target = st.selectbox(
                    "Which column says whether the customer churned?",
                    cols, index=guess_target_index(cols), key=f"custom_target_{uploaded.name}",
                )
                values = sorted(raw[target].dropna().unique().tolist(), key=str)

                if len(values) != 2:
                    st.error(
                        f"'{target}' has {len(values)} different values, but a churn column needs exactly 2 "
                        "(for example Yes/No or 1/0). Pick a different column."
                    )
                else:
                    positive = st.selectbox(
                        "Which value means the customer churned?",
                        values, index=guess_positive_index(values), format_func=str,
                        key=f"custom_pos_{uploaded.name}_{target}",
                    )
                    suggested = auto_ignore(coerce_numeric_like(raw), target)
                    ignore = st.multiselect(
                        "Columns to ignore (IDs, names, and anything recorded after the customer left)",
                        [c for c in cols if c != target], default=suggested,
                        key=f"custom_ignore_{uploaded.name}_{target}",
                    )
                    if suggested:
                        st.caption("Pre-selected: columns that look like IDs, names or constants.")

                    settings = (uploaded.name, target, str(positive), tuple(ignore))

                    if st.button("🚀 Train and score", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Training and checking the model..."):
                                result = train_custom_model(raw, target, positive, ignore)
                            st.session_state.custom_result = {"settings": settings, "result": result}
                        except ValueError as e:
                            st.session_state.custom_result = None
                            st.error(str(e))
                        except Exception:
                            st.session_state.custom_result = None
                            st.error(
                                "Something went wrong while training. Check that the churn column and "
                                "ignored columns are set correctly, then try again."
                            )

                    saved = st.session_state.get("custom_result")
                    if saved and saved["settings"] != settings:
                        st.info("You changed the settings. Click **Train and score** to update the results.")
                    elif saved:
                        res = saved["result"]
                        m = res["metrics"]
                        st.markdown("---")
                        st.subheader("📊 How well does the model work?")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
                        c2.metric("Recall (churners)", f"{m['recall']:.2f}")
                        c3.metric("Precision (churners)", f"{m['precision']:.2f}")
                        c4.metric("Accuracy", f"{m['accuracy']:.2f}")
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Churn rate in your file", f"{m['churn_rate']:.0%}")
                        d2.metric("Accuracy of always guessing 'no churn'", f"{m['baseline_accuracy']:.0%}")
                        d3.metric("Churn rate in the riskiest 10%", f"{m['top10_churn_rate']:.0%}",
                                  delta=f"{m['top10_lift']:.1f}× the average", delta_color="off")
                        st.caption(
                            "ROC-AUC: 0.5 is a coin flip and 1.0 is perfect. Every customer was scored by a model that "
                            f"never saw them (5-fold cross-validation, {m['rows']:,} customers, {res['n_features']} columns used)."
                        )
                        if m["churn_rate"] < 0.3 and m["recall"] < 0.3:
                            st.caption(
                                "Precision and recall use a 0.5 cut-off. When churn is rare they look low even for a useful model, "
                                "so the 'riskiest 10%' number above is the better guide."
                            )
                        if m["roc_auc"] < 0.6:
                            st.warning("This file doesn't seem to contain much signal about churn. "
                                       "The scores are only slightly better than guessing.")

                        if res["leaky"]:
                            names = ", ".join(f"**{c}** (AUC {a:.2f})" for c, a in res["leaky"])
                            st.warning(
                                f"Suspiciously strong column(s): {names}. A single column that predicts churn this well "
                                "is often recorded *after* the customer left, which makes the model look better than it "
                                "would be in real life. Add them to **Columns to ignore** above and train again for a "
                                "more realistic result."
                            )

                        st.markdown("---")
                        st.subheader("🔍 What goes with churn in this data?")
                        w1, w2 = st.columns(2)
                        with w1:
                            st.markdown("**Linked to higher churn**")
                            for name in res["drivers_up"] or ["Nothing notable"]:
                                st.write(f"🔺 {name}")
                        with w2:
                            st.markdown("**Linked to lower churn**")
                            for name in res["drivers_down"] or ["Nothing notable"]:
                                st.write(f"🔻 {name}")
                        st.caption("These are patterns in this file, not proven causes.")

                        st.markdown("---")
                        st.subheader("📋 Scored customers")
                        scored = res["scored"]
                        tiers = scored["risk_tier"].value_counts()
                        t1, t2, t3 = st.columns(3)
                        t1.metric("High risk (top 10%)", int(tiers.get("High", 0)))
                        t2.metric("Moderate (next 20%)", int(tiers.get("Moderate", 0)))
                        t3.metric("Low (the rest)", int(tiers.get("Low", 0)))
                        st.dataframe(scored.head(CUSTOM_PREVIEW_ROWS), use_container_width=True)
                        st.caption(
                            f"Showing the {min(CUSTOM_PREVIEW_ROWS, len(scored))} highest-risk customers. "
                            "The download has everyone. Risk tiers compare customers within this file."
                        )
                        st.download_button(
                            "⬇ Download scored file",
                            scored.to_csv(index=False).encode("utf-8"),
                            file_name="orbitly_custom_scored.csv", mime="text/csv",
                        )

        st.markdown("---")
        if st.button("⬅ Back", use_container_width=True, key="custom_back"):
            st.session_state.page = "welcome"
            st.rerun()