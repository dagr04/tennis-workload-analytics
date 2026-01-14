from workload_generator import WorkloadGenerator
from ml_models import build_features, split_data, train_random_forest
from coach_planner import display_coach_report


def main():
  
    generator = WorkloadGenerator(
        player_name="Alex",
        start_date="2022-01-01",
        days=1000
    )

    df = generator.run()

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    rf_model, rf_pred, rf_prob, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)

    print(f"\nRandom Forest trained successfully (ROC-AUC = {rf_auc:.3f})")

    display_coach_report(df, rf_model)

if __name__ == "__main__":
    main()
