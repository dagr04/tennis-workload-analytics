from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def build_features(df):
    features = [ 
        "Acute",
        "Chronic",
        "ACWR",
        "Intensity",
        "Workload",
        "HighIntensityStreak",
        "Fatigue"
    ]

    X = df[features]
    y = df["Injury"]

    return X, y


def split_data(X, y, ratio=0.7):
    split = int(len(X) * ratio)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]  

    lr_auc = roc_auc_score(y_test, lr_prob)

    return lr_model, lr_pred, lr_prob, lr_auc


def train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    rf_auc = roc_auc_score(y_test, rf_prob)
    

    return rf_model, rf_pred, rf_prob, rf_auc

def plot_roc_curves(y_test, lr_prob, rf_prob):
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], label='Random Guess (AUC = 0.50)')
    plt.xlabel('False Alarms (False Positive Rate)')
    plt.ylabel('Injuries Caught (True Positive Rate)')
    plt.title('Injury Prediction Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def compare_models(data):
    print("INJURY PREDICTION MODEL COMPARISON")

    X, y = build_features(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_model, lr_pred, lr_prob, lr_auc = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    rf_model, rf_pred, rf_prob, rf_auc = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    print("\n" + "-" * 80)
    print("LOGISTIC REGRESSION RESULTS")
    print(classification_report(y_test, lr_pred))
    print(f"ROC-AUC Score: {lr_auc:.4f}")
    print("\n" + "-" * 80)

    print("RANDOM FOREST RESULTS")
    print(classification_report(y_test, rf_pred))
    print(f"ROC-AUC Score: {rf_auc:.4f}")
    print("\n" + "-" * 80)

    print("MODEL COMPARISON SUMMARY")
    print(f"{'Metric':<25} {'Logistic Regression':<25} {'Random Forest':<25}")
    print(f"{'ROC-AUC Score':<25} {lr_auc:<25.4f} {rf_auc:<25.4f}")
    print("\n" + "-" * 80)

    winner = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
    print(f"\n WINNER: {winner} (Higher ROC-AUC)")

    plot_roc_curves(y_test, lr_prob, rf_prob)

    return lr_model, rf_model
