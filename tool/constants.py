#!/usr/bin/env python

"""Constants used in this project."""

INPUT_DIR = "data/input/"
INTERMEDIATE_DIR = "data/intermediate/"
OUTPUT_DIR = "data/output/"

UCDB_FILE_PATH = INPUT_DIR + "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
CA_TRACT_FILE_PATH = INPUT_DIR + "tl_2022_06_tract/tl_2022_06_tract.shp"
UBER_TRACT_FILE_PATH = INPUT_DIR + "los_angeles_censustracts.json"
UBER_TRAVEL_TIME_FILE_PATH = INPUT_DIR + "los_angeles-censustracts-2020-1-All-HourlyAggregate.csv"

LA_CLIP_CONVEX_FILE_PATH = INTERMEDIATE_DIR + "la_clip_convex.shp"
LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH = (
    INTERMEDIATE_DIR + "LA_clip_convex_strong_network_non_simplify_all_direction_working.graphml"
)

LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH_2023 = (
    INTERMEDIATE_DIR + "LA_clip_convex_strong_network_non_simplify_all_direction_2023.graphml"
)
LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH_2025 = (
    INTERMEDIATE_DIR + "LA_clip_convex_strong_network_non_simplify_all_direction_2025.graphml"
)

CONVEX_STRONGLY_ATTRIBUTES_FILE_PATH = (
    INTERMEDIATE_DIR + "nodes_candidate_convex_strongly_attributes.csv"
)
CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH = (
    INTERMEDIATE_DIR + "nodes_candidate_convex_strongly_attributes.pickle"
)
SAMPLED_OD_FILE_PATH = INTERMEDIATE_DIR + "OD_5m_strong.csv"
SAMPLED_OD_ALL_UBER_FILE_PATH = INTERMEDIATE_DIR + "OD_pairs_uber_all_strongly_119w.csv"
SAMPLED_OD_SAMPLE_HOUR_FILE_PATH = INTERMEDIATE_DIR + "OD3am.csv"
SAMPLED_OD_SAMPLE_HOUR_PICKLE_PATH = INTERMEDIATE_DIR + "OD3am.pickle"

SAMPLED_OD_ROUTES_API_FILE_PATH = INTERMEDIATE_DIR + "OD3am_routes_api.csv"
NETWORK_ROUTING_RESULT_FILE_PATH = (
    INTERMEDIATE_DIR + "OD3am_routes_api_network_routing_multiple_all.csv"
)

MAP_PATH = INTERMEDIATE_DIR + "test_train_split.csv"

TRAFFIC_CONTROL_FILE_PATH = OUTPUT_DIR + "table1_traffic_controls.csv"
COMBINED_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "table2_combined_evaluation_result.csv"

BEST_RF_RANDOM_PARAM_FILE_PATH = OUTPUT_DIR + "best_rf_random_param.csv"
BEST_GB_RANDOM_PARAM_FILE_PATH = OUTPUT_DIR + "best_gb_random_param.csv"
BEST_AB_RANDOM_PARAM_FILE_PATH = OUTPUT_DIR + "best_ab_random_param.csv"
BEST_DT_RANDOM_PARAM_FILE_PATH = OUTPUT_DIR + "best_dt_random_param.csv"

RF_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "rf_evaluation_result.csv"
GB_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "gb_evaluation_result.csv"
AB_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "ab_evaluation_result.csv"
DT_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "dt_evaluation_result.csv"
NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "naive_evaluation_result.csv"

SHAP_BEESWARM = OUTPUT_DIR + "rf_beeswarm_full.png"
SHAP_IMPORTANCE = OUTPUT_DIR + "rf_feature_importance_full.png"
SHAP_STATS = OUTPUT_DIR + "rf_shap_summary_stats_full.csv"

LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH = (
    INTERMEDIATE_DIR
    + "LA_clip_convex_strong_network_non_simplify_all_direction_working.graphml"
)

TRAFFIC_CONTROL_FILE_PATH_2025 = OUTPUT_DIR + "table1_traffic_controls_2025.csv"
NETWORK_ROUTING_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "network_routing_result_2025.csv"

RF_EVALUATION_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "rf_evaluation_result_2025.csv"
GB_EVALUATION_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "gb_evaluation_result_2025.csv"
DT_EVALUATION_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "dt_evaluation_result_2025.csv"
AB_EVALUATION_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "ab_evaluation_result_2025.csv"
NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH_2025 = OUTPUT_DIR + "naive_evaluation_result_2025.csv"
COMBINED_EVALUATION_RESULT_FILE_PATH_2025 = (
    OUTPUT_DIR + "table3_combined_evaluation_result_2025.csv"
)

BEST_RF_RANDOM_PARAM_FILE_PATH_2025 = OUTPUT_DIR + "best_rf_random_param_2025.csv"

# robustness 2025
SHAP_BEESWARM_2025 = OUTPUT_DIR + "shap_beeswarm_2025.png"
SHAP_IMPORTANCE_2025 = OUTPUT_DIR + "shap_importance_2025.png"
SHAP_STATS_2025 = OUTPUT_DIR + "shap_stats_2025.csv"
