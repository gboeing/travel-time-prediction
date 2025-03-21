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
LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH = INTERMEDIATE_DIR + "la_clip_convex_strong_network.graphml"
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
NETWORK_ROUTING_RESULT_FILE_PATH = INTERMEDIATE_DIR + "OD3am_routes_api_network_routing.csv"

PREDICTION_RESULT_FILE_PATH = OUTPUT_DIR + "prediction_result.csv"
RF_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "rf_evaluation_result.csv"
NETWORK_ROUTING_EVALUATION_RESULT_FILE_PATH = OUTPUT_DIR + "naive_evaluation_result.csv"
CROSS_VALIDATION_RESULT_FILE_PATH = OUTPUT_DIR + "cross_validation_result.txt"
BEST_RANDOM_PARAM_FILE_PATH = OUTPUT_DIR + "best_random_param.csv"
