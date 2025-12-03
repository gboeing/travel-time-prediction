#!/usr/bin/env python
"""Constants used in this project."""

from os import getenv
from pathlib import Path

YEAR = getenv("TTP_YEAR", "2023")

# Base Directories
INPUT_DIR = Path("data/input")
INTERMEDIATE_DIR = Path("data/intermediate")
OUTPUT_DIR = Path("data/output")

YEAR_INTERMEDIATE_DIR = INTERMEDIATE_DIR / YEAR
YEAR_OUTPUT_DIR = OUTPUT_DIR / YEAR

YEAR_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
YEAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Inputs
UCDB_FILE_PATH = INPUT_DIR / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
CA_TRACT_FILE_PATH = INPUT_DIR / "tl_2022_06_tract/tl_2022_06_tract.shp"
UBER_TRACT_FILE_PATH = INPUT_DIR / "los_angeles_censustracts.json"
UBER_TRAVEL_TIME_FILE_PATH = INPUT_DIR / "los_angeles-censustracts-2020-1-All-HourlyAggregate.csv"

# Intermediate common
LA_CLIP_CONVEX_FILE_PATH = INTERMEDIATE_DIR / "la_clip_convex.shp"
LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH = (
    INTERMEDIATE_DIR / "LA_clip_convex_strong_network_non_simplify_all_direction_working.graphml"
)
ROUTING_NETWORK_READ_PATH = (
    YEAR_INTERMEDIATE_DIR
    / f"LA_clip_convex_strong_network_non_simplify_all_direction_{YEAR}.graphml"
)

CONVEX_STRONGLY_ATTRIBUTES_FILE_PATH = (
    INTERMEDIATE_DIR / "nodes_candidate_convex_strongly_attributes.csv"
)
CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH = (
    INTERMEDIATE_DIR / "nodes_candidate_convex_strongly_attributes.pickle"
)
SAMPLED_OD_FILE_PATH = INTERMEDIATE_DIR / "OD_5m_strong.csv"
SAMPLED_OD_ALL_UBER_FILE_PATH = INTERMEDIATE_DIR / "OD_pairs_uber_all_strongly_119w.csv"
SAMPLED_OD_SAMPLE_HOUR_FILE_PATH = INTERMEDIATE_DIR / "OD3am.csv"
SAMPLED_OD_SAMPLE_HOUR_PICKLE_PATH = INTERMEDIATE_DIR / "OD3am.pickle"

SAMPLED_OD_ROUTES_API_FILE_PATH = INTERMEDIATE_DIR / "OD3am_routes_api.csv"
NETWORK_ROUTING_RESULT_FILE_PATH = YEAR_INTERMEDIATE_DIR / "network_routing_result.csv"

ROUTING_RESULT_FILE_PATH = YEAR_INTERMEDIATE_DIR / "network_routing_result.csv"
TRAFFIC_CONTROL_FILE_PATH = YEAR_OUTPUT_DIR / "table1_traffic_controls.csv"

# Modeling Results
NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "naive_evaluation_result.csv"
RF_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "rf_evaluation_result.csv"
GB_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "gb_evaluation_result.csv"
DT_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "dt_evaluation_result.csv"
AB_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "ab_evaluation_result.csv"
COMBINED_EVALUATION_RESULT_FILE_PATH = YEAR_OUTPUT_DIR / "combined_evaluation_result.csv"

# Modeling Params & Plots
BEST_RF_RANDOM_PARAM_FILE_PATH = YEAR_OUTPUT_DIR / "best_rf_random_param.csv"
BEST_GB_RANDOM_PARAM_FILE_PATH = YEAR_OUTPUT_DIR / "best_gb_random_param.csv"
BEST_DT_RANDOM_PARAM_FILE_PATH = YEAR_OUTPUT_DIR / "best_dt_random_param.csv"
BEST_AB_RANDOM_PARAM_FILE_PATH = YEAR_OUTPUT_DIR / "best_ab_random_param.csv"

SHAP_BEESWARM = YEAR_OUTPUT_DIR / "rf_beeswarm.png"
SHAP_IMPORTANCE = YEAR_OUTPUT_DIR / "rf_feature_importance.png"
SHAP_STATS = YEAR_OUTPUT_DIR / "rf_shap_summary_stats.csv"

# Split Map Logic
if YEAR == "2023":
    # 2023: 定义在自己的文件夹里
    MAP_PATH = YEAR_INTERMEDIATE_DIR / "test_train_split.csv"
    CREATE_SPLIT_MAP = True
else:
    # 2025 (或其他): 强制读取 2023 的 map
    MAP_PATH = INTERMEDIATE_DIR / "2023" / "test_train_split.csv"
    CREATE_SPLIT_MAP = False
