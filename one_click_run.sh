#!/bin/zsh

#conda env create -f environment.yml
conda activate ttp
YEARS=("2023" "2025")
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --requery)
      echo "requery the data..."
      python3 tool/01-preprocess_input_data.py
      python3 tool/02-routes_api.py "$2" "$3"
      shift 3 ;;
    --year)
      if [ "$2" = "all" ]; then YEARS=("2023" "2025"); else YEARS=("$2"); fi
      shift 2 ;;
    *) shift ;;
  esac
done

for YEAR in "${YEARS[@]}"; do
  echo "---------------------------------"
  echo "Processing Year: $YEAR"

  export TTP_YEAR="$YEAR"

  echo "routing and modeling..."
  python3 tool/03-network_routing.py
  python3 tool/04-modeling.py
done
