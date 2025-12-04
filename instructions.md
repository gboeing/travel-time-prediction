# Instructions on how to reproduce reported findings

These instructions provide step-by-step guidance to reproduce findings reported in the paper, including
- **Table 1. Traffic control elements summary**
- **Table 2. Out-of-sample prediction accuracy of our chosen models**
- **SHAP result for selected random forest model**

The workflow now supports multiple network years (e.g., 2023 Baseline and 2025 Robustness).

Change the current working directory to this folder, and in the terminal run `source one_click_run.sh`.

## Default Option: "One-Click Run" (Recommended)
To reproduce the reported results for **both years (2023 & 2025)**, open a terminal in this directory and run: `source one_click_run.sh`

(Optional) To run a specific year: `source one_click_run.sh --year 2023` or `source one_click_run.sh --year 2025`

**Fast Test Mode:**
To verify the code runs without errors using a small subset of data (runs in ~10 minute, all result would have a _test suffix):
`source one_click_run.sh --test`

**Model Only Mode:**
If you already have the routing results from step 3 OD routing (`03-network_routing.py`)and only want to retrain models (`04-modeling.py`):
`source one_click_run.sh --model-only`

This option will
- By default, bypass step 1 & 2 (`01-preprocess_input_data.py` and `02-routes_api.py`) and use instead our pre-queried intermediate data of
  - Drivable street network from OpenStreetMap (OSM)
    - `data/intermediate/2023/LA_clip_convex_strong_network_non_simplify_all_direction_2023.graphml`
    - `data/intermediate/2025/LA_clip_convex_strong_network_non_simplify_all_direction_2025.graphml`
  - Travel times from Google Routes API, for training the model.
    - `data/intermediate/OD3am_routes_api.csv`

- Then execute step 3 OD routing (`03-network_routing.py`) and step 4 modeling (`04-modeling.py`).
- #### Step 3 OD routing (`03-network_routing.py`)
  - Generates routing features (five kinds of traffic controls and turn counts along and route and edge traversal travel times) for all OD pairs sampled in the preprocessed OSM street network.
    - Output: `data/intermediate/<YEAR>/network_routing_result.csv`
  - Generates **Table 1. Traffic control elements summary**
    - Output: `data/output/<YEAR>/table1_traffic_controls.csv`
  - Approximate runtime: ~8 hours on 10 cores

- #### Step 4 modeling (`04-modeling.py`)
  - Chooses the best performing hyperparameters for four machine learning models: random forest, decision trees, adaboost, and gradient boosting, and evaluates their prediction accuracy
    - Output: `data/output/<YEAR>/best_rf_random_param.csv` (and gb, ab, dt...)
  - Evaluates these models' prediction accuracy. **Note: 2025 robustness check uses the fixed train/test split from 2023.**
    - Output: `data/output/<YEAR>/rf_evaluation_result.csv` (and gb, ab, dt...)
  - Aggregates results to generate **Table 2. Out-of-sample prediction accuracy**
    - Output: `data/output/<YEAR>/table2_combined_evaluation_result.csv`
  - Approximate runtime: ~1 hour

## Alternative Option: Full Pipeline Run-through with Re-query (NOT Recommended)

This option will
- Rerun the entire workflow from processing raw input data and re-querying street network from OSM and travel time from Google Routes API: `source one_click_run.sh --requery your_own_google_api_key 2025-05-01`
- Replace `your_own_google_api_key`with your Google API Key
- Replace `2025-05-01` with any date in the future to query travel time at 3 am.

We do not recommend this option for reproducibility because:
  - First, it may provide different results compared with our study, as the numbers surely change slightly from one day to another.
  - Second, you need to bring an API key to query Google Routes API.

### Step 1 pre-process input data (`01-preprocess_input_data.py`)
  - Preprocesses the input (prediction) data, re-queries drivable street network from OSM (will differ from ours), and re-samples OD pairs (will differ from our paper).
### Step 2 query Google routes' API's travel time (`02-routes_api.py`)
  - Re-queries Google Routes API for (training) travel times.
  - **Important:** The script will **automatically create a `data/intermediate/working/` folder** and move the newly generated network files there.
  - The pipeline will then proceed to run Step 3 (Routing) and Step 4 (Modeling) exclusively on this new **"working"** dataset, creating a new train/test split automatically.

## Folder Structure
```text
.
├── data
│   ├── input
│   │   ├── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg                                 # Global Human Settlement Layer’s
│   │   ├── los_angeles_censustracts.json                                              # LA county uber movement tract boundary
│   │   ├── los_angeles-censustracts-2020-1-All-HourlyAggregate.csv                    # Uber movement data
│   │   └── tl_2022_06_tract
│   ├── intermediate
│   │   ├── la_clip_convex.shp
│   │   ├── OD3am_routes_api.csv
│   │   ├── 2023
│   │   │   ├── LA_clip_convex_strong_network_non_simplify_all_direction_2023.graphml  # 2023 network
│   │   │   ├── network_routing_result.csv                                             # Generated by Step 3
│   │   │   └── test_train_split.csv                                                   # Master split map
│   │   ├── 2025
│   │   │   ├── LA_clip_convex_strong_network_non_simplify_all_direction_2025.graphml  # 2025 network
│   │   │   └── network_routing_result.csv
│   │   └── working                                                                    # Generated if --requery is used
│   │       ├── LA_clip_convex_strong_network_non_simplify_all_direction_working.graphml
│   │       └── ...
│   ├── output
│   │   ├── 2023
│   │   │   ├── best_rf_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── best_gb_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── best_ab_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── best_dt_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── rf_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── gb_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── ab_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── dt_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   │   ├── table1_traffic_controls.csv                                                # TO BE generated by 03-network_routing.py (Table 1)
│   │   │   └── table2_combined_evaluation_result.csv
│   │   └── 2025
│   │       └── ... (same structure as 2023)
│   │   └── working
│   │       └── ... (same structure as 2023)
│   └── README.md
├── environment.yml
├── instructions.md
├── one_click_run.sh
├── README.md
└── tool
    ├── 01-preprocess_input_data.py
    ├── 02-routes_api.py
    ├── 03-network_routing.py
    ├── 04-modeling.py
    └── constants.py
