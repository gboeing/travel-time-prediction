# Data

The input training data can be accessed here (restricted access due to source terms of use and rehosting limitations):
[here](https://drive.google.com/drive/folders/1G444vNZN7TvW5C5Dw9VC_KBqy-iBaXQX?usp=sharing)

It is organized according to the structure listed in the constants.py file.
There are 3 folders: `input`, `intermediate`, and `output`.
The `input` folder contains the all the data. Theoretically, we should be able to run the entire pipeline with just the
data in the `input` folder.
However, our project's existing results are constrained by and based on:
1. Google travel time we queried in the early 2024, which is the `intermediate/OD3am_routes_api.csv` file,
2. Street network we queried in the late 2023, which is the `intermediate/la_clip_convex_strong_network.graphml` file.

The `output` folder contains the results of the analysis pipeline.
