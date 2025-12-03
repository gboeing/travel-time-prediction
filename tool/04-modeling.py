#!/usr/bin/env python

"""Building random forest model to predict Google travel time and compute SHAP explanations."""

import logging
from pathlib import Path

import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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


def read_result(ff_path: str) -> pd.DataFrame:
    """Read in and combine the edge traversal time routing result and Google Route API result.

    Returns
    -------
        pd.DataFrame: Combined result of edge traversal time routing and Google Route API.

    """
    ff_result = pd.read_csv(ff_path)
    google_result = pd.read_csv(constants.SAMPLED_OD_ROUTES_API_FILE_PATH)
    return ff_result.merge(google_result, left_on=["oid", "did"], right_on=["oid", "did"])


def initial_random_split_and_save_map(
    df_og: pd.DataFrame,
    map_path: str = constants.MAP_PATH,
    test_size: float = 0.2,
    random_state: int = 123,
) -> None:
    """Create an initial random train/test split on the baseline data and save the mapping.

    This function performs a one-time random split of the baseline (2023) dataset into
    training and test sets, then saves a mapping from each (oid, did) pair to its
    assigned set ("train" or "test"). The saved mapping can be reused later to ensure
    that other datasets (e.g., 2025 network results) use the exact same split.

    Parameters
    ----------
    df_og : pd.DataFrame
        2023 combined dataset (routing results joined with Google travel times).
    map_path : str, optional
        File path where the split mapping CSV will be saved.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Random seed used for the initial train/test split, by default 123.

    """
    train, test = train_test_split(df_og, test_size=test_size, random_state=random_state)
    split_map = pd.concat([train.assign(_set="train"), test.assign(_set="test")])[
        ["oid", "did", "_set"]
    ].drop_duplicates(subset=["oid", "did"])
    split_map.to_csv(map_path, index=False)


def split_with_saved_map(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    map_path: str = constants.MAP_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train and test sets using a saved split mapping.

    This function applies a precomputed train/test assignment to a new dataset.
    Only rows OD pairs in the mapping are retained, ensuring a consistent split.

    Parameters
    ----------
    df : pd.DataFrame
        Combined result dataset.
    features : list of str
        List of column names to use as model features.
    target : str
        Column name to use as the prediction target.
    map_path : str, optional
        File path to the CSV mapping with columns ["oid", "did", "_set"].

    Returns
    -------
    x_train : pd.DataFrame
        Feature matrix for the training set.
    y_train : pd.Series
        Target values for the training set.
    x_test : pd.DataFrame
        Feature matrix for the test set.
    y_test : pd.Series
        Target values for the test set.
    train_ids : pd.DataFrame
        OD pairs in the training set.
    test_ids : pd.DataFrame
        OD pairs in the test set.

    """
    split_map = pd.read_csv(map_path)
    df_new = df.merge(split_map, on=["oid", "did"], how="inner")

    dropped = len(df) - len(df_new)
    if dropped > 0:
        logger = logging.getLogger("tool")
        logger.warning("Dropped %d rows not in saved split map.", dropped)

    train_mask = df_new["_set"] == "train"
    test_mask = df_new["_set"] == "test"

    x_train = df_new.loc[train_mask, features]
    y_train = df_new.loc[train_mask, target]
    x_test = df_new.loc[test_mask, features]
    y_test = df_new.loc[test_mask, target]

    train_ids = df_new.loc[train_mask, ["oid", "did"]]
    test_ids = df_new.loc[test_mask, ["oid", "did"]]

    return x_train, y_train, x_test, y_test, train_ids, test_ids


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
    x_train, y_train, x_test, y_test, _, _ = split_with_saved_map(
        df,
        features=_FEATURES_LIST,
        target=_PREDICT_VARIABLE,
        map_path=constants.MAP_PATH,
    )
    return x_train, y_train, x_test, y_test


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


def random_forest(  # noqa: PLR0913 # many arguments by design
    df: pd.DataFrame,
    out_path: str,
    save_best_params_path: str,
    shap_beeswarm_path: str,
    shap_importance_path: str,
    shap_stats_path: str,
) -> None:
    """Build and tune a random forest model to predict travel time and compute SHAP.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.
    out_path : str
        File path where the Random Forest evaluation result CSV will be saved.
    save_best_params_path : str
        File path where the Random Forest parameter CSV will be saved.
    shap_beeswarm_path : str
        File path where the SHAP beeswarm plot (PNG) will be saved.
    shap_importance_path : str
        File path where the SHAP global feature importance bar plot (PNG) will be saved.
    shap_stats_path : str
        File path where the SHAP summary statistics CSV will be saved.


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
    pd.DataFrame([rf_random.best_params_]).to_csv(save_best_params_path)
    best_random = RandomForestRegressor(**rf_random.best_params_, random_state=123)

    best_random.fit(x_train, y_train)
    tuned_y_pred = best_random.predict(x_test)
    x_full = df[_FEATURES_LIST]
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

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(out_path, index=False)

    logger = logging.getLogger("tool")
    logger.info("Starting SHAP analysis for Random Forest...")

    # use training set as background to build tree SHAP explainer
    explainer = shap.TreeExplainer(
        best_random,
        data=x_train,
        feature_perturbation="interventional",
    )
    logger.info("Calculating SHAP values for full sample...")
    shap_values = explainer(x_full)
    logger.info("SHAP values for full sample calculated.")

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, x_full, show=False, cmap=plt.get_cmap("plasma"))
    plt.title("SHAP Beeswarm Plot (Random Forest Tuned, full sample)")
    plt.savefig(shap_beeswarm_path, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved to %s", shap_beeswarm_path)

    # Global feature importance bar plot
    plt.figure()
    shap.summary_plot(shap_values, x_full, plot_type="bar", show=False, cmap=plt.get_cmap("plasma"))
    plt.title("SHAP Global Feature Importance (Random Forest Tuned, full sample)")
    plt.savefig(shap_importance_path, bbox_inches="tight")
    plt.close()
    logger.info("SHAP feature importance plot saved to %s", shap_importance_path)

    # SHAP summary statistics
    logger.info("Calculating SHAP summary statistics for full sample...")
    try:
        shap_values_data = shap_values.values  # noqa: PD011 # shap_values is shap.Explanation, not pandas.DataFrame
    except AttributeError:
        shap_values_data = shap_values  # numpy array fallback

    shap_df = pd.DataFrame(shap_values_data, columns=x_full.columns)

    stats_df = pd.DataFrame(
        {
            "feature": x_full.columns,
            "mean_shap_value": shap_df.mean(),
            "median_shap_value": shap_df.median(),
            "mean_abs_shap_value": shap_df.abs().mean(),
            "min_shap_value": shap_df.min(),
            "max_shap_value": shap_df.max(),
        },
    ).sort_values(by="mean_abs_shap_value", ascending=False)

    stats_df.to_csv(shap_stats_path, index=False)
    logger.info("SHAP summary statistics saved to %s", shap_stats_path)


def gradient_boost(df: pd.DataFrame, out_path: str) -> None:
    """Build and tune a gradient boosting model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.
    out_path : str
        File path where the Gradient Boost evaluation result CSV will be saved.


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
    best_random = GradientBoostingRegressor(**gb_random.best_params_, random_state=123)
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

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(out_path, index=False)


def decision_tree(df: pd.DataFrame, out_path: str) -> None:
    """Build and tune a decision tree regression model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.
    out_path : str
        File path where the Decision Tree evaluation result CSV will be saved.


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
    pd.concat([base_evaluation, tuned_evaluation]).to_csv(out_path, index=False)


def adaboost(df: pd.DataFrame, out_path: str) -> None:
    """Build and tune an Adaboost regression model to predict Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.
    out_path : str
        File path where the Adaboost evaluation result CSV will be saved.


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

    pd.concat([base_evaluation, tuned_evaluation]).to_csv(out_path, index=False)


def naive(df: pd.DataFrame, out_path: str) -> None:
    """Evaluate naive travel time against Google travel time.

    Parameters
    ----------
    df: pd.DataFrame
        Combined result of naive travel time, turns and traffic controls, and Google travel time.
    out_path : str
        File path where the naive travel time evaluation result CSV will be saved.


    Returns
    -------
    None

    """
    x_train, y_train, x_test, y_test = dataset_split(df)

    evaluation = model_evaluation(y_test, x_test["travel_time"], "Naive")
    evaluation.to_csv(out_path, index=False)


def run_for_network(  # noqa: PLR0913 # many arguments by design
    ff_path: str,
    rf_out: str,
    gb_out: str,
    dt_out: str,
    ab_out: str,
    naive_out: str,
    combined_out: str,
    save_best_params_path: str,
    shap_beeswarm_path: str,
    shap_importance_path: str,
    shap_stats_path: str,
    *,
    create_split_map: bool = False,
) -> None:
    """Run the full modeling workflow for a single network.

    This function:
    1) reads the routing results and joins them with Google travel times;
    2) optionally creates a persistent train/test split map based on (oid, did);
    3) fits and evaluates all models ;
    4) writes per-model metrics and a combined evaluation table to disk;
    5) for the Random Forest model, computes SHAP explanations and saves plots/statistics.

    Parameters
    ----------
    ff_path : str
        File path to the routing results (naive travel time + features) for this network.
    rf_out : str
        Output path for Random Forest evaluation CSV.
    gb_out : str
        Output path for Gradient Boosting evaluation CSV.
    dt_out : str
        Output path for Decision Tree evaluation CSV.
    ab_out : str
        Output path for AdaBoost evaluation CSV.
    naive_out : str
        Output path for naive (baseline) evaluation CSV.
    combined_out : str
        Output path for the combined evaluation table CSV.
    save_best_params_path : str
        File path where the Random Forest parameter CSV will be saved.
    shap_beeswarm_path : str
        Output path for the SHAP beeswarm plot (Random Forest, PNG).
    shap_importance_path : str
        Output path for the SHAP feature importance plot (Random Forest, PNG).
    shap_stats_path : str
        Output path for the SHAP summary statistics CSV (Random Forest).
    create_split_map : bool, optional
        If True, create the initial train/test split mapping using this
        dataset and save it to constants.MAP_PATH.

    """
    logger = logging.getLogger("tool")

    result = read_result(ff_path)

    if create_split_map:
        map_path = Path(constants.MAP_PATH)
        if not map_path.exists():
            logger.info("Split map not found. Creating split map at %s", map_path)
            initial_random_split_and_save_map(result, constants.MAP_PATH)
        else:
            logger.info("Split map already exists at %s; using existing map.", map_path)

    logger.info("random_forest start...")
    random_forest(
        result,
        rf_out,
        save_best_params_path,
        shap_beeswarm_path,
        shap_importance_path,
        shap_stats_path,
    )
    logger.info("random_forest done.")

    logger.info("gradient_boost start...")
    gradient_boost(result, gb_out)
    logger.info("gradient_boost done.")

    logger.info("decision_tree start...")
    decision_tree(result, dt_out)
    logger.info("decision_tree done.")

    logger.info("adaboost start...")
    adaboost(result, ab_out)
    logger.info("adaboost done.")

    naive(result, naive_out)

    # combine evaluation result
    file_paths = [naive_out, rf_out, gb_out, dt_out, ab_out]
    df_list = [pd.read_csv(fp) for fp in file_paths]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.iloc[:, 1:-1]
    combined_df = combined_df[combined_df["Model"].str.contains("Tuned|^Naive$", regex=True)]
    combined_df["MAPE (%)"] = combined_df["MAPE (%)"] * 100
    combined_df["Model"] = combined_df["Model"].str.replace(" Tuned", "", regex=False)
    combined_df["Model"] = combined_df["Model"].replace({"Naive": "Initial naïve"})
    combined_df = combined_df.round(2)
    combined_df.to_csv(combined_out, index=False)


if __name__ == "__main__":
    logging.basicConfig(filename="tool.log", filemode="w", level=logging.INFO)
    logger = logging.getLogger("tool")
    logger.setLevel(logging.INFO)

    cv_logger = logging.getLogger("cv")
    cv_logger.setLevel(logging.WARNING)

    # --- 2023 baseline ---
    run_for_network(
        constants.NETWORK_ROUTING_RESULT_FILE_PATH,
        constants.RF_EVALUATION_RESULT_FILE_PATH,
        constants.GB_EVALUATION_RESULT_FILE_PATH,
        constants.DT_EVALUATION_RESULT_FILE_PATH,
        constants.AB_EVALUATION_RESULT_FILE_PATH,
        constants.NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH,
        constants.COMBINED_EVALUATION_RESULT_FILE_PATH,
        constants.BEST_RF_RANDOM_PARAM_FILE_PATH,
        constants.SHAP_BEESWARM,
        constants.SHAP_IMPORTANCE,
        constants.SHAP_STATS,
        create_split_map=True,
    )

    # --- 2025 robustness ---
    run_for_network(
        constants.NETWORK_ROUTING_RESULT_FILE_PATH_2025,
        constants.RF_EVALUATION_RESULT_FILE_PATH_2025,
        constants.GB_EVALUATION_RESULT_FILE_PATH_2025,
        constants.DT_EVALUATION_RESULT_FILE_PATH_2025,
        constants.AB_EVALUATION_RESULT_FILE_PATH_2025,
        constants.NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH_2025,
        constants.BEST_RF_RANDOM_PARAM_FILE_PATH_2025,
        constants.SHAP_BEESWARM_2025,
        constants.SHAP_IMPORTANCE_2025,
        constants.SHAP_STATS_2025,
        constants.COMBINED_EVALUATION_RESULT_FILE_PATH_2025,
    )
