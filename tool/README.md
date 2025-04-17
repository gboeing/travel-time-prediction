# Prediction Model

Run the model by executing the code in order:

1. run the `01-preprocess.py` file to preprocess the data and sample OD pairs.
2. run the `02-routes_api.py` file to query the 3am Google travel time of sampled OD pairs using Google Routes API.
3. run the `03-network_routing.py` file to generate edge traversal travel time and count turns and traffic controls
along the routing of the sampled OD pairs.
4. run the `04-random_forest.py` file to predict Google travel time based on the edge traversal travel time routing.
