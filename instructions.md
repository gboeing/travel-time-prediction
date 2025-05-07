# Instructions on how to reproduce reported findings "Travel Time Prediction from Sparse Open Data"

These instructions provide step-by-step guidance to reproduce findings reported in the paper, including
- **Table 1. Traffic control elements summary**
- **Table 2. Out-of-sample prediction accuracy of our chosen models**

Simply change the current working directory under this folder, and in the shell run `source one_click_run.sh`.

## Default Option: "One-Click Run" (Recommended)
To reproduce the reported results, open a terminal in this directory and run:
```bash
source one_click_run.sh
```
This option will
- By default, bypass step 1 & 2 (`01-preprocess_input_data.py` and `02-routes_api.py`) and use instead our pre-queried intermediate data of
  - Drivable street network from OpenStreetMap(OSM)
    - `data/intermediate/LA_clip_convex_strong_network_non_simplify_all_direction.graphml`
  - Travel times from Google Routes API
    - `data/intermediate/OD3am_routes_api.csv`

- Then execute step 3 OD routing (`03-network_routing.py`) and step 4 modeling (`04-modeling.py`).
- #### Step 3 OD routing (`03-network_routing.py`)
  - Generates routing features (five kinds of traffic controls and turn counts along and route and edge traversal travel times) for all OD pairs sampled in the preprocessed OSM street network.
    - Output: `data/intermediate/OD3am_routes_api_network_routing_multiple_all.csv`
  - Generates **Table 1. Traffic control elements summary**
    - Output: `data/output/table1_traffic_controls.csv`
  - Approximate runtime: ~8 hours on 10 cores

- #### Step 4 modeling (`04-modeling.py`)
  - Chooses the best performing hyperparameters for four machine learning models: random forest, decision trees, adaboost, and gradient boosting, and evaluates their prediction accuracy
    - Output: best performing hyperparameters for each of the four models:
      - Random forest: `data/output/best_rf_random_param.csv`
      - Gradient boosting: `data/output/best_gb_random_param.csv`
      - Adaboost: `data/output/best_ab_random_param.csv`
      - Decision trees: `data/output/best_dt_random_param.csv`
  - Evaluates these models' prediction accuracy
    - Output: evaluation results based on MAPE, MAE, difference-in-means, p-values, R-square, as well as the whole sample five-fold cross-validation results
      - Random forest: `data/output/rf_evaluation_result.csv`
      - Gradient boosting: `data/output/gb_evaluation_result.csv`
      - Adaboost: `data/output/ab_evaluation_result.csv`
      - Decision trees: `data/output/dt_evaluation_result.csv`
  - Aggregates results to generate **Table 2. Out-of-sample prediction accuracy**
    - Output: `data/output/table2_combined_evaluation_result.csv`
  - Approximate runtime: ~1 hour

## Alternative Option: Full Pipeline Run-through with Re-query (Not Recommended)

This option will
- Rerun the entire workflow from processing raw input data and re-querying street network from OSM and travel time from Google Routes API:

```bash
source one_click_run.sh --requery your_own_google_api_key 2024-02-01
```
- Replace`your_own_google_api_key`with your Google API Key
- Replace `2024-02-01` with any date in the future to query travel time at 3 am.
We do not recommend this option because:
  - First, it will provide different results compared with our study, as our results are based on the pre-queried OpenStreetMap and Google Routes API in late 2023 and early 2024.
  - Second, you need to bring an API key to query Google Routes API, and Google is likely to charge for the usage.

- #### Step 1 pre-process input data (`01-preprocess_input_data.py`)
  - Preprocesses the input data, re-queries drivable street network from OSM (will differ from ours), and re-samples OD pairs (will differ from our paper).
- #### Step 2 query Google routes' API's travel time (`02-routes_api.py`)
  - Re-queries Google Routes API for travel times (requires your Google API key which may incur costs, and the result will differ from ours due to different query times).

## Folder Structure
```text
.
├── data
│   ├── input                                                                          #raw input datasets
│   │   ├── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg                                 # Global Human Settlement Layer’s
│   │   ├── los_angeles_censustracts.json                                              # LA county uber movement tract boundary
│   │   ├── los_angeles-censustracts-2020-1-All-HourlyAggregate.csv                    # Uber movement data
│   │   └── tl_2022_06_tract                                                           # LA county census tract boundary
│   ├── intermediate
│   │   ├── LA_clip_convex_strong_network_non_simplify_all_direction.graphml           # pre-queried street network
│   │   ├── OD3am_routes_api.csv                                                       # pre-queried sampled OD pairs with Google travel time
│   │   └──OD3am_routes_api_network_routing_multiple_all.csv                           # TO BE generated by 03-network_routing.py
│   ├── output                                                                         #FOLDER WHERE ALL THE OUTPUS ARE
│   │   ├── best_rf_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── best_gb_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── best_ab_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── best_dt_random_param.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── rf_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── gb_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── ab_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── dt_evaluation_result.csv                                                   # TO BE generated by 04-modeling.py
│   │   ├── table1_traffic_controls.csv                                                # TO BE generated by 03-network_routing.py (Table 1)
│   │   └── table2_combined_evaluation_result.csv                                      # TO BE generated by 04-modeling.py (Table 2)

│   └── README.md
├── environment.yml
├── instructions.md                                                                    # This file
├── one_click_run.sh
├── README.md
└── tool
    ├── 01-preprocess_input_data.py
    ├── 02-routes_api.py
    ├── 03-network_routing.py
    ├── 04-modeling.py
    ├── constants.py
    └── README.md
