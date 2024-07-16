import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def prepare_data():
    with mlflow.start_run(run_name="Data Preparation",nested=True):
        # Load data
        housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")

        # Create income categories for stratified sampling
        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., float('inf')],
                                       labels=[1, 2, 3, 4, 5])

        # Stratified sampling
        strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

        # Drop the income_cat column to return to the original data
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        mlflow.log_param("train_size", len(strat_train_set))
        mlflow.log_param("test_size", len(strat_test_set))

        return strat_train_set, strat_test_set

if __name__ == "__main__":
    prepare_data()
