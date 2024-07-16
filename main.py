import mlflow
import prepare_data
import train_model
import score_model

def main():
    with mlflow.start_run(run_name="Main Run") as main_run:
        # Data Preparation
        with mlflow.start_run(run_name="Data Preparation", nested=True):
            strat_train_set, strat_test_set = prepare_data.prepare_data()
            strat_train_set.to_csv('strat_train_set.csv', index=False)
            strat_test_set.to_csv('strat_test_set.csv', index=False)

        # Model Training
        with mlflow.start_run(run_name="Model Training", nested=True):
            train_model.train_model()

        # Model Scoring
        with mlflow.start_run(run_name="Model Scoring", nested=True):
            score_model.score_model()

if __name__ == "__main__":
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow")
    main()
