"""Constants used in this project."""

INPUT_DIR = "data/input/"
INTERMEDIATE_DIR = "data/intermediate/"

UCDB_FILE_PATH = INPUT_DIR + "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
CA_TRACT_FILE_PATH = INPUT_DIR + "tl_2022_06_tract/tl_2022_06_tract.shp"
UBER_TRACT_FILE_PATH = INPUT_DIR + "los_angeles_censustracts.json"

LA_CLIP_CONVEX_FILE_PATH = INTERMEDIATE_DIR + "la_clip_convex.shp"
LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH = INTERMEDIATE_DIR + "la_clip_convex_strong_network.graphml"
CONVEX_STRONGLY_ATTRIBUTES_FILE_PATH = (
    INTERMEDIATE_DIR + "nodes_candidate_convex_strongly_attributes.csv"
)
