Simply change the current working directory under this folder, and in the shell run `source one_click_run.sh`.

- It will by default bypass step 1 & 2 (`01-preprocess_input_data.py` and `02-routes_api.py`)and use our pre-queried intermediate data of drivable street network from OpenStreetMap and travel times from Google Routes API.
- Step 3 OD routing (`03-network_routing.py`)may take around 8 hours and step 4 modeling (`04-modeling.py`) may take around 1 hour.
- If you'd prefer, you could also directly run step 4 modeling using script `04-modeling.py`.

If you don't want to bypass step 1 and 2, in the shell run `source one_click_run.sh --requery your_own_google_api_key 2024-02-01`

- Running step 1 and 2 will re-query the OpenStreetMap and Google Routes API travel time.
- Replace your_own_google_api_key with your Google API Key, and replace 2024-02-01 with any date in the future that you wish to query travel time at 3 am.
- We advise you against running step 1 & 2 because: First, it will provide different results compared with our study, because our results are based on the pre-queried OpenStreetMap and Google Routes API in late 2023 and early 2024. Second, you will also need bring an API key to query Google Routes API, and Google is likely to charge you for the queries.
