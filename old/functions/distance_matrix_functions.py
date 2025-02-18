import json
import os
from datetime import datetime

import googlemaps
import pytz

# api key
key = "Jessica_Key"
gmaps = googlemaps.Client(key=key)


def cache_distance_info(origin, destination, data=None):
    """Cache distance information for a specific origin-destination pair, or retrieve it if already cached.

    Parameters
    ----------
    origin : str
        A string representing the starting point of the route.
    destination : str
        A string representing the destination point of the route.
    data : dict, optional
        The distance information to be cached. If provided, the function will save this data to the cache.
        If None, the function attempts to retrieve cached data for the given origin-destination pair.

    Returns
    -------
    dict or None
        If `data` is None and a cache file exists for the specified route, this function returns the cached data.
        If `data` is None and no cache file exists, or if `data` is provided, it returns None.

    Notes
    -----
    This function handles both writing to and reading from a cache, depending on whether `data` is provided.

    """
    cache_dir = "distance_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = os.path.join(cache_dir, f"distance_{origin}_to_{destination}.json")

    if data is not None:
        with open(cache_file, "w") as file:
            json.dump(data, file)
        return None

    if os.path.exists(cache_file):
        with open(cache_file) as file:
            return json.load(file)
    return None


def batch_distance_requests(origins, destinations, mode, batch_size, departure_time=None):
    """Perform batched distance matrix requests between origins and destinations, utilizing caching for efficiency.

    Parameters
    ----------
    origins : list
        A list of origins, where each element can be a string or a tuple representing the geographic coordinates.
    destinations : list
        A list of destinations, corresponding to the origins, formatted similarly.
    mode : str
        The mode of transportation for the distance calculation.
    batch_size : int
        The number of origin-destination pairs to process in each batch.
    departure_time : tuple of int, optional
        The departure time for the routes, represented as a tuple (year, month, day, hour, minute, second).
        If None, the current time is used.

    Returns
    -------
    dict
        A dictionary where each key is a tuple (origin, destination) and the value is the response data from the
        distance matrix request or the cached data.

    """
    results = {}

    # convert timezone
    if departure_time == None:
        departure_timestamp = "now"
    else:
        departure_timestamp = None
        if departure_time:
            local_tz = pytz.timezone("US/Pacific")
            departure_time = datetime(*departure_time, tzinfo=local_tz)
            departure_time_utc = departure_time.astimezone(pytz.utc)
            departure_timestamp = int(departure_time_utc.timestamp())

    for i in range(0, len(origins), batch_size):
        batch_origins = origins[i : i + batch_size]
        batch_destinations = destinations[i : i + batch_size]

        for origin, destination in zip(batch_origins, batch_destinations, strict=False):
            cache_key = (origin, destination)
            cached_data = cache_distance_info(*cache_key)

            if cached_data:
                results[cache_key] = cached_data
            else:
                # request
                response = gmaps.distance_matrix(
                    origin,
                    destination,
                    mode=mode,
                    departure_time=departure_timestamp,
                )
                cache_distance_info(*cache_key, data=response)
                results[cache_key] = response

    return results
