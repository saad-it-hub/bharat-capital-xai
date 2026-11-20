from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "indian_fintech_loans.csv"
MODEL = BASE / "loan_decision_tree.joblib"

FEATURES = [
    "city", "employment_type", "cibil_score", "annual_income_inr",
    "loan_amount_inr", "dti_percentage", "recent_defaults"
]
NUMERIC = ["cibil_score", "annual_income_inr", "loan_amount_inr", "dti_percentage", "recent_defaults"]

def train_model():
    df = pd.read_csv(DATA)
    X = pd.get_dummies(df[FEATURES], columns=["city", "employment_type"])
    y = df["loan_approved"].astype(int)
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump({"model": model, "columns": list(X.columns), "features": FEATURES}, MODEL)
    return accuracy

def load_model():
    if not MODEL.exists():
        train_model()
    return joblib.load(MODEL)

def encode_applicant(applicant, columns):
    row = pd.DataFrame([{
        "city": applicant["city"],
        "employment_type": applicant["employment_type"],
        "cibil_score": applicant["cibil_score"],
        "annual_income_inr": applicant["annual_income_inr"],
        "loan_amount_inr": applicant["loan_amount_inr"],
        "dti_percentage": applicant["dti_percentage"],
        "recent_defaults": applicant.get("recent_defaults", 0)
    }])
    row = pd.get_dummies(row, columns=["city", "employment_type"])
    return row.reindex(columns=columns, fill_value=0)

def explain_prediction(model, encoded_row, columns, approved):
    """Walk the exact decision-tree path and translate split conditions to readable text."""
    tree = model.tree_
    node = 0
    reasons = []
    values = encoded_row.iloc[0].to_dict()
    while tree.children_left[node] != tree.children_right[node]:
        feature_index = tree.feature[node]
        feature = columns[feature_index]
        threshold = tree.threshold[node]
        value = float(values.get(feature, 0))
        went_left = value <= threshold
        if feature in ("cibil_score", "annual_income_inr", "loan_amount_inr", "dti_percentage", "recent_defaults"):
            labels = {
                "cibil_score": "CIBIL score",
                "annual_income_inr": "annual income",
                "loan_amount_inr": "requested loan amount",
                "dti_percentage": "DTI",
                "recent_defaults": "recent defaults"
            }
            pretty = labels[feature]
            if feature in ("annual_income_inr", "loan_amount_inr"):
                threshold_text = f"₹{threshold:,.0f}"
            elif feature == "dti_percentage":
                threshold_text = f"{threshold:.1f}%"
            else:
                threshold_text = f"{threshold:.1f}"
            op = "<=" if went_left else ">"
            condition = f"{pretty} {op} {threshold_text}"
        else:
            category = feature.split("_", 1)[1].replace("_", " ")
            is_match = value == 1
            condition = f"{category.title()} {'matches' if is_match else 'does not match'} {feature.split('_')[0].replace('employment', 'employment type').title()}"
        reasons.append(condition)
        node = tree.children_left[node] if went_left else tree.children_right[node]
    if approved:
        return "Approved by the model based on the applicant profile."
    return "The model's decision path: " + " AND ".join(reasons) + ". These are model-based factors, not a guarantee of a lender's final decision."
