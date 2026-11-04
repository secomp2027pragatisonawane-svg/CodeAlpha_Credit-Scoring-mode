"""
Credit Scoring Model
--------------------
Predicts whether an applicant is a bad credit risk (default = 1) from past
financial data using Logistic Regression, Decision Tree and Random Forest.

To use your own data, replace `generate_data()` with a pandas read_csv call that
returns a DataFrame with the same columns (see COLUMNS below) and a binary
`default` target.
"""cd 
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             accuracy_score, confusion_matrix, roc_curve,
                             classification_report)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
SEED = 42


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def generate_data(n=20000):
    """Synthetic credit history data with a realistic default mechanism."""
    age = RNG.integers(21, 70, n)
    income = np.exp(RNG.normal(10.9, 0.55, n)).round(-2)            # annual income
    employment_years = np.clip(age - 21 - RNG.integers(0, 10, n), 0, None)
    credit_history_years = np.clip(age - 18 - RNG.integers(0, 12, n), 0, None)
    loan_amount = (income * RNG.uniform(0.1, 1.2, n)).round(-2)
    monthly_debt = (income / 12 * RNG.beta(2, 5, n) * 0.9).round(0)
    credit_limit = (income * RNG.uniform(0.2, 1.0, n)).round(-2)
    credit_used = (credit_limit * RNG.beta(2, 3, n)).round(0)
    num_open_accounts = RNG.poisson(5, n) + 1
    late_30 = RNG.poisson(0.6, n)
    late_60 = RNG.poisson(0.25, n)
    late_90 = RNG.poisson(0.1, n)
    num_inquiries = RNG.poisson(1.5, n)
    home_ownership = RNG.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.4, 0.15])
    loan_purpose = RNG.choice(["debt_consolidation", "home", "car", "education", "other"],
                              n, p=[0.35, 0.2, 0.2, 0.1, 0.15])

    # Latent risk -> default probability
    dti = monthly_debt * 12 / income
    util = credit_used / credit_limit
    lti = loan_amount / income
    z = (-3.2 + 2.8 * dti + 1.8 * util + 0.9 * lti
         + 0.35 * late_30 + 0.6 * late_60 + 0.9 * late_90
         + 0.12 * num_inquiries
         - 0.03 * credit_history_years - 0.03 * employment_years
         + 0.4 * (home_ownership == "RENT") - 0.3 * (home_ownership == "OWN")
         + 0.3 * (loan_purpose == "other")
         + RNG.normal(0, 0.6, n))
    p = 1 / (1 + np.exp(-z))
    default = (RNG.uniform(size=n) < p).astype(int)

    df = pd.DataFrame(dict(
        age=age, income=income, employment_years=employment_years,
        credit_history_years=credit_history_years, loan_amount=loan_amount,
        monthly_debt=monthly_debt, credit_limit=credit_limit, credit_used=credit_used,
        num_open_accounts=num_open_accounts, late_30=late_30, late_60=late_60,
        late_90=late_90, num_inquiries=num_inquiries, home_ownership=home_ownership,
        loan_purpose=loan_purpose, default=default))

    # Inject realistic missing values
    for col, frac in [("employment_years", 0.04), ("income", 0.02), ("credit_used", 0.03)]:
        df.loc[df.sample(frac=frac, random_state=SEED).index, col] = np.nan
    return df


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer_features(df, medians=None):
    df = df.copy()
    num_cols = ["income", "employment_years", "credit_used"]
    if medians is None:                       # fit on train only -> no leakage
        medians = df[num_cols].median()
    for c in num_cols:
        df[c + "_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(medians[c])

    df["debt_to_income"] = df["monthly_debt"] * 12 / df["income"]
    df["credit_utilization"] = (df["credit_used"] / df["credit_limit"]).clip(0, 1.5)
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["payment_burden"] = df["monthly_debt"] / (df["income"] / 12)
    df["total_late_payments"] = df["late_30"] + df["late_60"] + df["late_90"]
    df["weighted_delinquency"] = df["late_30"] + 2 * df["late_60"] + 3 * df["late_90"]
    df["has_severe_delinquency"] = (df["late_90"] > 0).astype(int)
    df["avg_credit_per_account"] = df["credit_limit"] / df["num_open_accounts"]
    df["history_to_age"] = df["credit_history_years"] / df["age"]
    df["log_income"] = np.log1p(df["income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    df["inquiries_per_year"] = df["num_inquiries"] / (df["credit_history_years"] + 1)

    df = pd.get_dummies(df, columns=["home_ownership", "loan_purpose"], drop_first=True)
    return df, medians


# ----------------------------------------------------------------------------
# 3. MODELS & EVALUATION
# ----------------------------------------------------------------------------
def evaluate(name, model, Xte, yte, threshold=0.5):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    return dict(Model=name,
                Accuracy=accuracy_score(yte, pred),
                Precision=precision_score(yte, pred),
                Recall=recall_score(yte, pred),
                F1=f1_score(yte, pred),
                ROC_AUC=roc_auc_score(yte, proba)), proba, pred


def main():
    df = generate_data()
    print(f"Dataset: {df.shape[0]} rows, default rate = {df['default'].mean():.1%}")

    train_raw, test_raw = train_test_split(df, test_size=0.2, stratify=df["default"],
                                           random_state=SEED)
    Xtr, medians = engineer_features(train_raw.drop(columns="default"))
    Xte, _ = engineer_features(test_raw.drop(columns="default"), medians)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    ytr, yte = train_raw["default"], test_raw["default"]

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, probas, preds = [], {}, {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        m.fit(Xtr, ytr)
        r, p, pr = evaluate(name, m, Xte, yte)
        r["CV_ROC_AUC"] = cv_auc
        rows.append(r); probas[name] = p; preds[name] = pr

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\n=== Test-set performance ===")
    print(results.to_string())
    results.to_csv("model_comparison.csv")

    best = results["ROC_AUC"].idxmax()
    print(f"\nBest model by ROC-AUC: {best}")
    print(classification_report(yte, preds[best], target_names=["Good", "Default"]))

    # ---- Plots ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax[0].legend(loc="lower right")

    cm = confusion_matrix(yte, preds[best])
    ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Good", "Default"],
              yticklabels=["Good", "Default"], xlabel="Predicted", ylabel="Actual",
              title=f"Confusion Matrix - {best}")
    plt.tight_layout(); plt.savefig("roc_and_confusion.png", dpi=140); plt.close()

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values().tail(15)
    plt.figure(figsize=(8, 6))
    imp.plot.barh(color="#2b6cb0")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=140); plt.close()
    print("\nTop 10 features:\n", imp.sort_values(ascending=False).head(10).round(4).to_string())

    # ---- Threshold analysis (business trade-off) ----
    p = probas[best]
    print(f"\n=== Threshold trade-off ({best}) ===")
    print("threshold  precision  recall   f1")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pr = (p >= t).astype(int)
        print(f"   {t:.1f}      {precision_score(yte, pr):.3f}     {recall_score(yte, pr):.3f}   {f1_score(yte, pr):.3f}")

    # ---- Score a new applicant ----
    applicant = pd.DataFrame([dict(
        age=35, income=900000, employment_years=8, credit_history_years=10,
        loan_amount=400000, monthly_debt=20000, credit_limit=300000, credit_used=90000,
        num_open_accounts=4, late_30=0, late_60=0, late_90=0, num_inquiries=1,
        home_ownership="RENT", loan_purpose="car")])
    Xa, _ = engineer_features(applicant, medians)
    Xa = Xa.reindex(columns=Xtr.columns, fill_value=0)
    pd_ = models[best].predict_proba(Xa)[0, 1]
    print(f"\nSample applicant -> probability of default: {pd_:.1%}")


if __name__ == "__main__":
    main()
