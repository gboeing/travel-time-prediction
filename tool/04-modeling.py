#!/usr/bin/env python

"""Building random forest model to predict Google travel time."""

import logging

import constants
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

_FEATURES_LIST = [
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
_PREDICT_VARIABLE = "duration"


def read_result() -> pd.DataFrame:
    """Read in and combine the edge traversal time routing result and Google Route API result.

    Returns
    -------
        pd.DataFrame: Combined result of edge traversal time routing and Google Route API.

    """
    ff_result = pd.read_csv(constants.NETWORK_ROUTING_RESULT_FILE_PATH)
    google_result = pd.read_csv(constants.SAMPLED_OD_ROUTES_API_FILE_PATH)
    return ff_result.merge(google_result, left_on=["oid", "did"], right_on=["oid", "did"])


def dataset_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split 80% train and 20% test set.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.

    Returns
    -------
    a tuple of pandas dataframes and pandas series of test set and train set.

    """
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    return (
        train[_FEATURES_LIST],
        train[_PREDICT_VARIABLE],
        test[_FEATURES_LIST],
        test[_PREDICT_VARIABLE],
    )


def model_evaluation(y: pd.Series, predictions: pd.Series, model_name: str) -> pd.DataFrame:
    """Evaluate the model prediction accuracy.

    Parameters
    ----------
    y: pd.Series
        The actual travel time.
    predictions: pd.Series
        The predicted travel time.
    model_name: str
        The name of the model to evaluate, this is annotated in the data frame returned.

    Returns
    -------
        a tuple of float including MSE, MAE, R2 score, MAPE, and accuracy.

    """
    p_value = stats.ttest_rel(y, predictions).pvalue
    apr = np.mean(predictions / y)
    diff = np.mean(predictions) - np.mean(y)

    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    return pd.DataFrame(
        [(model_name, mape, mae, mse, diff, p_value, apr, r2)],
        columns=["Model", "MAPE (%)", "MAE", "MSE", "Mean Diff", "p-value", "APR", "R² Score"],
    )


def random_forest(df: pd.DataFrame) -> None:
    """Build and tune a random forest model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.


    Returns
    -------
        None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    # Fitting the default Random forest Regression to the dataset
    rf = RandomForestRegressor(random_state=123)
    # Fit the regressor with x and y data
    rf.fit(x_train, y_train)
    base_y_pred = rf.predict(x_test)
    base_evaluation = model_evaluation(y_test, base_y_pred, "Random Forest Base")
    base_evaluation["cross_val_score"] = [
        cross_val_score(
            rf,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    random_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        "min_samples_split": [2, 5, 10],
        "bootstrap": [True, False],
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
    rf_random.fit(x_train, y_train)
    pd.DataFrame([rf_random.best_params_]).to_csv(constants.BEST_RF_RANDOM_PARAM_FILE_PATH)
    best_random = RandomForestRegressor(**rf_random.best_params_)

    best_random.fit(x_train, y_train)
    tuned_y_pred = best_random.predict(x_test)
    tuned_evaluation = model_evaluation(y_test, tuned_y_pred, "Random Forest Tuned")
    tuned_evaluation["cross_val_score"] = [
        cross_val_score(
            best_random,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(constants.RF_EVALUATION_RESULT_FILE_PATH)


def gradient_boost(df: pd.DataFrame) -> None:
    """Build and tune a gradient boosting model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.


    Returns
    -------
    None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    # Fitting the default gradient boosting Regression to the dataset
    gb = GradientBoostingRegressor(random_state=123)
    # Fit the regressor with x and y data
    gb.fit(x_train, y_train)
    base_y_pred = gb.predict(x_test)
    base_evaluation = model_evaluation(y_test, base_y_pred, "Gradient Boosting Base")
    base_evaluation["cross_val_score"] = [
        cross_val_score(
            gb,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    random_grid = {
        "loss": ["squared_error", "absolute_error", "huber", "quantile"],
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "subsample": np.linspace(0.6, 1.0, 5),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_depth": range(5, 11),
        "max_features": ["sqrt", "log2", None],
        "warm_start": [True, False],
    }
    gb_random = RandomizedSearchCV(
        estimator=gb,
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        random_state=123,
        n_jobs=-1,
        return_train_score=True,
    )
    gb_random.fit(x_train, y_train)
    pd.DataFrame([gb_random.best_params_]).to_csv(constants.BEST_GB_RANDOM_PARAM_FILE_PATH)
    best_random = GradientBoostingRegressor(**gb_random.best_params_)
    best_random.fit(x_train, y_train)
    tuned_y_pred = best_random.predict(x_test)
    tuned_evaluation = model_evaluation(y_test, tuned_y_pred, "Gradient Boosting Tuned")
    tuned_evaluation["cross_val_score"] = [
        cross_val_score(
            best_random,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(constants.GB_EVALUATION_RESULT_FILE_PATH)


def decision_tree(df: pd.DataFrame) -> None:
    """Build and tune a decision tree regression model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.


    Returns
    -------
    None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    # Fitting the default Decision Tree Regression to the dataset
    dt = DecisionTreeRegressor(random_state=123)
    # Fit the regressor with x and y data
    dt.fit(x_train, y_train)
    base_y_pred = dt.predict(x_test)
    base_evaluation = model_evaluation(y_test, base_y_pred, "Decision Tree Base")
    base_evaluation["cross_val_score"] = [
        cross_val_score(
            dt,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    random_grid = {
        "max_features": ["sqrt", "log2", None],
        "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10, 20],
    }
    dt_random = RandomizedSearchCV(
        estimator=dt,
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        random_state=123,
        n_jobs=-1,
        return_train_score=True,
    )
    dt_random.fit(x_train, y_train)
    pd.DataFrame([dt_random.best_params_]).to_csv(constants.BEST_DT_RANDOM_PARAM_FILE_PATH)

    best_random = DecisionTreeRegressor(**dt_random.best_params_, random_state=123)
    best_random.fit(x_train, y_train)
    tuned_y_pred = best_random.predict(x_test)
    tuned_evaluation = model_evaluation(y_test, tuned_y_pred, "Decision Tree Tuned")
    tuned_evaluation["cross_val_score"] = [
        cross_val_score(
            best_random,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]
    pd.concat([base_evaluation, tuned_evaluation]).to_csv(constants.DT_EVALUATION_RESULT_FILE_PATH)


def adaboost(df: pd.DataFrame) -> None:
    """Build and tune an Adaboost regression model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.


    Returns
    -------
    None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    # Fitting the default AdaBoost Regression to the dataset
    base_dt = DecisionTreeRegressor(random_state=123)
    ab = AdaBoostRegressor(estimator=base_dt, random_state=123)
    ab.fit(x_train, y_train)
    base_y_pred = ab.predict(x_test)
    base_evaluation = model_evaluation(y_test, base_y_pred, "AdaBoost Base")
    base_evaluation["cross_val_score"] = [
        cross_val_score(
            ab,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    random_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "learning_rate": [0.001, 0.01, 0.1, 1.0, 10],
        "loss": ["linear", "square", "exponential"],
        "estimator__max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 5, 10, 20],
    }

    ab_random = RandomizedSearchCV(
        estimator=ab,
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        random_state=123,
        n_jobs=-1,
        return_train_score=True,
    )
    ab_random.fit(x_train, y_train)
    pd.DataFrame([ab_random.best_params_]).to_csv(constants.BEST_AB_RANDOM_PARAM_FILE_PATH)

    best_base_dt = DecisionTreeRegressor(
        max_depth=ab_random.best_params_["estimator__max_depth"],
        min_samples_split=ab_random.best_params_["estimator__min_samples_split"],
        min_samples_leaf=ab_random.best_params_["estimator__min_samples_leaf"],
        random_state=123,
    )
    best_random = AdaBoostRegressor(
        estimator=best_base_dt,
        n_estimators=ab_random.best_params_["n_estimators"],
        learning_rate=ab_random.best_params_["learning_rate"],
        loss=ab_random.best_params_["loss"],
        random_state=123,
    )
    best_random.fit(x_train, y_train)
    tuned_y_pred = best_random.predict(x_test)
    tuned_evaluation = model_evaluation(y_test, tuned_y_pred, "AdaBoost Tuned")
    tuned_evaluation["cross_val_score"] = [
        cross_val_score(
            best_random,
            df[_FEATURES_LIST],
            df[_PREDICT_VARIABLE],
            cv=kf,
            scoring="neg_mean_absolute_error",
        ),
    ]

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(constants.AB_EVALUATION_RESULT_FILE_PATH)


def naive(df: pd.DataFrame) -> None:
    """Evaluate native travel time against Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.


    Returns
    -------
    None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)

    evaluation = model_evaluation(y_test, x_test["travel_time"], "Naive")
    evaluation.to_csv(constants.NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH)


if __name__ == "__main__":
    logger = logging.getLogger("tool")
    logging.basicConfig(filename="tool.log", filemode="w", level=logging.INFO)
    logger.setLevel(logging.INFO)

    cv_logger = logging.getLogger("cv")
    cv_logger.setLevel(logging.WARNING)

    result = read_result()

    logger.info("random_forest start...")
    random_forest(result)
    logger.info("random_forest done.")
    logger.info("gradient_boost start...")
    gradient_boost(result)
    logger.info("gradient_boost done.")
    logger.info("decision_tree start...")
    decision_tree(result)
    logger.info("decision_tree done.")
    logger.info("adaboost start...")
    adaboost(result)
    logger.info("adaboost done.")
    naive(result)

    # combine evaluation result
    file_paths = [
        constants.NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH,
        constants.RF_EVALUATION_RESULT_FILE_PATH,
        constants.GB_EVALUATION_RESULT_FILE_PATH,
        constants.DT_EVALUATION_RESULT_FILE_PATH,
        constants.AB_EVALUATION_RESULT_FILE_PATH,
    ]
    df_list = [pd.read_csv(fp) for fp in file_paths]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.iloc[:, 1:-1]
    combined_df = combined_df[combined_df["Model"].str.contains("Tuned|^Naive$", regex=True)]
    combined_df["MAPE (%)"] = combined_df["MAPE (%)"] * 100
    combined_df["Model"] = combined_df["Model"].str.replace(" Tuned", "", regex=False)
    combined_df["Model"] = combined_df["Model"].replace({"Naive": "Initial naïve"})
    combined_df = combined_df.round(2)
    combined_df.to_csv(constants.COMBINED_EVALUATION_RESULT_FILE_PATH)
