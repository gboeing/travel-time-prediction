#!/bin/zsh

# Activate the conda environment
#conda env create -f environment.yml
conda activate ttp

# 1. Reset test flag to ensure clean state for every run
export TTP_TEST=""

# 2. Default settings
YEARS=("2023" "2025")
RUN_ROUTING=true

# 3. Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    # --- Option A: Re-query Data (Full Pipeline) ---
    # Usage: source one_click_run.sh --requery <API_KEY> <DATE>
    --requery)
      echo ">>> Requerying data (Step 1 & 2)..."
      python3 tool/01-preprocess_input_data.py
      python3 tool/02-routes_api.py "$2" "$3"

      echo ">>> Data requery done. Switching to 'working' mode."
      YEARS=("working")

      shift 3 ;;

    # --- Option B: Select Specific Year ---
    # Usage: source one_click_run.sh --year 2023
    --year)
      if [ "$2" = "all" ]; then YEARS=("2023" "2025"); else YEARS=("$2"); fi
      shift 2 ;;

    # --- Option C: Test Mode (Fast Run) ---
    # Usage: source one_click_run.sh --test
    --test)
      export TTP_TEST="true"
      shift ;;

    # --- Option D: Model Only (Skip Routing) ---
    # Usage: source one_click_run.sh --model-only
    --model-only)
      RUN_ROUTING=false
      shift ;;

    *) shift ;;
  esac
done

# 4. Main Execution Loop
for YEAR in "${YEARS[@]}"; do
  echo "Processing Year: $YEAR"

  if [ "$TTP_TEST" = "true" ]; then
      echo "TEST MODE: ON (Data truncated, _test suffixes added)"
  fi

  # Set environment variable for Python scripts (constants.py reads this)
  export TTP_YEAR="$YEAR"

  # Step 1: Network Routing (Skip if --model-only is set)
  if [ "$RUN_ROUTING" = true ]; then
      echo ">> Step 1: Routing (03)..."
      python3 tool/03-network_routing.py
  else
      echo ">> Skipping Routing (Model Only)..."
  fi

  # Step 2: Modeling
  echo ">> Step 2: Modeling (04)..."
  python3 tool/04-modeling.py

  echo "Finished Year: $YEAR"
done

echo ""
echo "All tasks finished."
