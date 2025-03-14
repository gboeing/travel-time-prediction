# Code files

All code files go in this folder.

Add any instructions/notes on running things to this readme file.

The code order is as follows:

1. run the `preprocess.py` file to preprocess the data and sample OD pairs.
2. run the `routes_api.py` file to query the 3am Google travel time of sampled OD pairs using Google Routes API.
3. run the `network_routing.py` file to generate edge traversal travel time and count turns and traffic controls
along the routing of the sampled OD pairs.
4. run the `random_forest.py` file to predict Google travel time based on the edge traversal travel time routing.
