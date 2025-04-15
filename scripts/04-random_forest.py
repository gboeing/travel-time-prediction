#!/usr/bin/env python

"""Building random forest model to predict Google travel time."""

from pathlib import Path

import constants
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    KFold
)


def read_result() -> pd.DataFrame:
    """Read in and combine the edge traversal time routing result and Google Route API result.

    Returns
    -------
        pd.DataFrame: Combined result of edge traversal time routing and Google Route API.

    """
    ff_result = pd.read_csv(constants.NETWORK_ROUTING_RESULT_FILE_PATH)
    google_result = pd.read_csv(constants.SAMPLED_OD_ROUTES_API_FILE_PATH)
    return ff_result.merge(google_result, left_on=["oid", "did"], right_on=["oid", "did"])


def rf_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a random forest model to predict Google travel time.

    Returns
    -------
        a pd.Dataframe of the testing set with a column of baseline random forest prediction and
        a column of hyper-tuned random forest prediction, a np.ndarray cross validation score of
        baseline and hyper-tuned random forest prediction.

    """
    # split 80% train and 20% test set
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    list_of_features = [
        "signal_count",
        "stop_count",
        "crossing_count",
        "give_way_count",
        "mini_roundabout_count",
        "left_count",
        "slight_left_count",
        "right_count",
        "slight_right_count",
        "u_count",
        "travel_time",
    ]
    y = train["duration"]
    x = train[list_of_features]
    x_test = test[list_of_features]
    # Fitting the default Random forest Regression to the dataset
    rf = RandomForestRegressor(random_state=123)
    # Fit the regressor with x and y data
    rf.fit(x, y)
    # Predict the result
    test["rf_predict_default"] = rf.predict(x_test)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    base_cross_val_score = cross_val_score(
        rf,
        df[list_of_features],
        df["duration"],
        cv=kf,
        scoring="neg_mean_absolute_error",
    )

    # Create and randomized grid of hyper parameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ["sqrt", "log2", None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, None]
    # Minimum number of samples required at each leaf node
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "bootstrap": bootstrap,
    }
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        random_state=123,
        n_jobs=-1,
        return_train_score=True,
    )
    rf_random.fit(x, y)
    best_random_param = rf_random.best_params_
    pd.DataFrame([best_random_param]).to_csv(constants.BEST_RANDOM_PARAM_FILE_PATH)
    best_random = RandomForestRegressor(**best_random_param)
    best_random.fit(x, y)
    # Predict the result
    test["rf_predict_random_tuned"] = best_random.predict(x_test)
    tuned_cross_val_score = cross_val_score(
        best_random,
        df[list_of_features],
        df["duration"],
        cv=kf,
        scoring="neg_mean_absolute_error",
    )
    test.to_csv(constants.PREDICTION_RESULT_FILE_PATH)

    # Write the results to a text file
    with Path.open(constants.CROSS_VALIDATION_RESULT_FILE_PATH, "w") as file:
        file.write(f"Base Cross-Validation Score: {base_cross_val_score}\n")
        file.write(f"Tuned Cross-Validation Score: {tuned_cross_val_score}\n")

    return test, pd.DataFrame([best_random_param]), base_cross_val_score, tuned_cross_val_score


def model_evaluation(y: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    """Evaluate the model prediction accuracy.

    Parameters
    ----------
    y: pd.Series
        The actual travel time.
    predictions: pd.Series
        The predicted travel time.

    Returns
    -------
        a tuple of float including MSE, R2 score, MAPE, and accuracy.

    """
    ttest_result = stats.ttest_rel(y, predictions)
    p_value = ttest_result.pvalue
    apr = np.mean(predictions / y)
    diff = np.mean(predictions) - np.mean(y)
    r2 = r2_score(y, predictions)
    errors = abs(y - predictions)
    mae = np.mean(errors)
    mape = 100 * np.mean(errors / y)
    return pd.DataFrame(
        {
            "Metric": ["MAPE (%)", "MAE", "Mean Diff", "p-value", "APR", "R² Score"],
            "Value": [mape, mae, diff, p_value, apr, r2],
        },
    )


if __name__ == "__main__":
    result = read_result()
    test1, best_params_df, base_cross_val_score, tuned_cross_val_score = rf_model(result)
    rf_eval = model_evaluation(test1["duration"], test1["rf_predict_random_tuned"])
    rf_eval.to_csv(constants.RF_EVALUATION_RESULT_FILE_PATH)
    naive_eval = model_evaluation(test1["duration"], test1["travel_time"])
    naive_eval.to_csv(constants.NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH)
