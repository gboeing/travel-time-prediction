#!/bin/zsh

#conda env create -f environment.yml
conda activate ttp

if [ "$1" = "--requery" ]
  then echo "requery the data..."
  python3 tool/01-preprocess_input_data.py
  python3 tool/02-routes_api.py "$2" "$3"
fi

echo "routing and modeling..."
python3 tool/03-network_routing.py
python3 tool/04-modeling.py
