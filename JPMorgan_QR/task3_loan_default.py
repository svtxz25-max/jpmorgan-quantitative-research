import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Task_3_and_4_Loan_Data.csv")
df = df.drop(columns=["customer_id"])

features = ["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding",
            "income", "years_employed", "fico_score"]

df["debt_to_income"] = df["total_debt_outstanding"] / df["income"]
df["loan_to_income"] = df["loan_amt_outstanding"] / df["income"]
features += ["debt_to_income", "loan_to_income"]

X = df[features]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

models = {"Logistic Regression": (lr, X_test_scaled), "Random Forest": (rf, X_test), "Gradient Boosting": (gb, X_test)}

print("=" * 55)
print("         MODEL COMPARISON (AUC-ROC)")
print("=" * 55)
for name, (model, X_eval) in models.items():
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
axes[0].barh([features[i] for i in sorted_idx], importances[sorted_idx], color="steelblue")
axes[0].set_title("Random Forest Feature Importances")
axes[0].set_xlabel("Importance")

from sklearn.metrics import roc_curve
for name, (model, X_eval) in models.items():
    proba = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

RECOVERY_RATE = 0.10
LOSS_GIVEN_DEFAULT = 1 - RECOVERY_RATE

def expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                  income, years_employed, fico_score):

    debt_to_income = total_debt_outstanding / income
    loan_to_income = loan_amt_outstanding / income

    input_df = pd.DataFrame([[
        credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
        income, years_employed, fico_score, debt_to_income, loan_to_income
    ]], columns=features)

    pd_lr = lr.predict_proba(scaler.transform(input_df))[0][1]
    pd_rf = rf.predict_proba(input_df)[0][1]
    pd_gb = gb.predict_proba(input_df)[0][1]
    pd_ensemble = (pd_lr + pd_rf + pd_gb) / 3

    el = pd_ensemble * LOSS_GIVEN_DEFAULT * loan_amt_outstanding

    print("=" * 50)
    print("         EXPECTED LOSS REPORT")
    print("=" * 50)
    print(f"  Loan Amount           : ${loan_amt_outstanding:>12,.2f}")
    print(f"  Recovery Rate         : {RECOVERY_RATE*100:.0f}%")
    print(f"  Loss Given Default    : {LOSS_GIVEN_DEFAULT*100:.0f}%")
    print("-" * 50)
    print(f"  PD (Logistic Reg)     : {pd_lr*100:>10.2f}%")
    print(f"  PD (Random Forest)    : {pd_rf*100:>10.2f}%")
    print(f"  PD (Gradient Boost)   : {pd_gb*100:>10.2f}%")
    print(f"  PD (Ensemble Avg)     : {pd_ensemble*100:>10.2f}%")
    print("=" * 50)
    print(f"  EXPECTED LOSS         : ${el:>12,.2f}")
    print("=" * 50)

    return round(el, 2)

print("\n--- TEST 1: High risk borrower ---")
expected_loss(5, 5000, 25000, 40000, 1, 540)

print("\n--- TEST 2: Low risk borrower ---")
expected_loss(0, 3000, 2000, 90000, 7, 750)

print("\n--- TEST 3: Mid-range borrower ---")
expected_loss(2, 4500, 10000, 65000, 4, 630)