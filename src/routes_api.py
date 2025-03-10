import concurrent.futures
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import polyline
import pytz
import requests

_CACHE_FOLDER = "data/route_cache"
os.makedirs(_CACHE_FOLDER, exist_ok=True)


class Location:
    """This class stores the longitude, latitude, and address of a geographic location.

    Parameters
    ----------
    lon : float
        The longitude coordinate of the location.
    lat : float
        The latitude coordinate of the location.
    addr : str
        The address or description of the location.

    """

    def __init__(self, lon, lat, addr):
        self.lon = lon
        self.lat = lat
        self.addr = addr


def parse_dataframe_to_locations(dataframe: pd.DataFrame) -> List[Location]:
    """Convert a DataFrame containing geometry and address information into a list of Location objects.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame containing geometry and address information.

    Returns
    -------
    locations : list of Location
        A list of Location objects representing geographic points with longitude, latitude, and address.

    Notes
    -----
    The input DataFrame should contain columns for geometry (in a specific format) and address information.
    The function parses the geometry information to extract longitude, latitude, and uses the provided address.

    """
    locations = []
    for i in range(len(dataframe)):
        point = str(dataframe.geometry.iloc[i])
        pattern = r"\((.*?)\)"
        longLat = re.findall(pattern, point)

        if longLat:
            longLat = longLat[0]
            splitIndex = longLat.find(" ")
            longitude = longLat[:splitIndex]
            latitude = longLat[splitIndex + 1:]

            # Check if the 'address' column exists and use it if available
            address = dataframe.address.iloc[i] if "address" in dataframe.columns else None
            locations.append(Location(longitude, latitude, address))

    return locations


def create_departure_time(departure_time: Tuple = None) -> str:
    """Convert a departure time to a standardized UTC format string.

    Parameters
    ----------
    departure_time : tuple, optional
        A tuple representing the year, month, day, hour, minute, and, second.
        For example, (2023, 11, 20, 8, 30, 0) represents the 20th of November 2023 at 8:30:00 am.
        If None, the function returns None, indicating no specific departure time.

    Returns
    -------
    str or None
        A string representing the departure time in UTC format ("%Y-%m-%dT%H:%M:%SZ").
        Returns None if no departure_time is provided or if there's an error in conversion.

    Notes
    -----
    The time is localized to the 'US/Pacific' timezone and converted to UTC.
    If the provided departure_time is invalid, the function will catch the exception and return None.

    """
    local_tz = pytz.timezone("US/Pacific")
    departure_time_str = None  # default value for departure_time_str

    if departure_time is not None:
        try:
            # Create a naive datetime object
            naive_departure_time = datetime(*departure_time)
            # Localize the datetime object to the specific timezone
            localized_departure_time = local_tz.localize(naive_departure_time)
            # Convert to UTC
            departure_time_utc = localized_departure_time.astimezone(pytz.utc)
            departure_time_str = departure_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            # Handle exceptions (like invalid departure_time format)
            print(f"Error: {e}")
            return None

    return departure_time_str


def calculate_route(apikey: str, orig: Location, dest: Location, traffic: str = "TRAFFIC_UNAWARE",
        departure_time: Tuple = None, ) -> Dict:
    """Calculate a route between two locations using the Google Routes API.

    Parameters
    ----------
    apikey : str
        The API key for accessing the Google Routes API.
    orig : Location
        Class object representing the origin of the route.
    dest : Location
        Class object representing the destination of the route.
    traffic : str
        Traffic awareness mode ('TRAFFIC_UNAWARE', 'TRAFFIC_AWARE', 'TRAFFIC_AWARE_OPTIMAL').
    departure_time : tuple
        A tuple representing the departure time in the format (year, month, day, hour, minute) (default=None).

    Returns
    -------
    dict
        A dictionary containing route information, including duration, distance, and encoded polyline.

    """
    # set up the headers and body for the request
    heads = {"X-Goog-Api-Key": apikey, "Content-Type": "application/json",
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline", }

    body = {"origin": {"location": {"latLng": {"latitude": orig.lat, "longitude": orig.lon, }, }, },
        "destination": {
            "location": {"latLng": {"latitude": dest.lat, "longitude": dest.lon, }, }, },
        "travelMode": "DRIVE", "routingPreference": traffic, }

    # add departureTime to the body if it is not None
    if departure_time is not None:
        body["departureTime"] = create_departure_time(departure_time)

    # send the request to the Google Routes API
    response = requests.post("https://routes.googleapis.com/directions/v2:computeRoutes",
        headers=heads, json=body, )
    return response.json()


def create_dataframe_from_results(route_results) -> pd.DataFrame:
    """Create a DataFrame from given route results.

    Parameters
    ----------
    route_results : list
        A list containing route information for each route entry.

    Returns
    -------
    routes_df : pandas.DataFrame
        A Pandas DataFrame containing route information, including distance, duration, and polyline.

    """
    # Initialize an empty list to store route data
    route_data = []

    # Iterate over the route results list
    for result in route_results:
        routes = result.get("routes", [])
        for route in routes:
            distance = route.get("distanceMeters", None)
            duration = route.get("duration", None)
            if duration is not None:
                duration = int(duration.rstrip("s"))  # Remove 's' and convert to int
            decoded_polyline = polyline.decode(
                route.get("polyline", {}).get("encodedPolyline", None), )

            # Append a dictionary of the route data to the route_data list
            route_data.append(
                {"distance": distance, "duration": duration, "polyline": decoded_polyline, }, )

    # Convert the route_data list of dictionaries to a DataFrame
    routes_df = pd.DataFrame(route_data)

    return routes_df


def cache_file_path(origin, destination, departure_time):
    """Generate a file path for caching route data based on the origin and destination coordinates.

    Parameters
    ----------
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.
    departure_time : tuple
        The departure time for the route, used to create a unique filename.

    Returns
    -------
    str
        A string representing the path to the cache file, including a unique filename based on the latitude,
        longitude of origin, destination, and departure time.

    Notes
    -----
    This function is used to create a standardized file path for caching route information.
    The file is saved with a '.json' extension and stored in a predefined cache folder.

    """
    departure_time_str = create_departure_time(departure_time)
    filename = f"route_{origin.lat}_{origin.lon}_to_{destination.lat}_{destination.lon}_{departure_time_str}.json"
    return os.path.join(_CACHE_FOLDER, filename)


def save_to_cache(data, origin, destination, departure_time):
    """Save route data to a cache file.

    Parameters
    ----------
    data : dict
        The data to be cached. This would be route information in a dictionary format.
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.
    departure_time : tuple
        A tuple representing the departure time, used in generating the cache filename.

    Returns
    -------
    None
        This function does not return a value. It performs the operation of writing data to a file.

    Notes
    -----
    This function saves route data to a cache file to avoid redundant computations or API calls in the future.

    """
    file_path = cache_file_path(origin, destination, departure_time)
    with Path.open(file_path, "w") as file:
        json.dump(data, file)


def load_from_cache(origin, destination, departure_time):
    """Load route data from a cache file, if it exists.

    Parameters
    ----------
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.
    departure_time : tuple
        A tuple representing the departure time, used in determining the cache file.

    Returns
    -------
    dict or None
        If a cached file exists for the specified route, the function returns the data in dictionary format.
        If no cache file exists for the route, the function returns None.

    Notes
    -----
    The function first checks if a cache file exists at the generated path using `os.path.exists`. If the file exists,
    the function opens the file in 'r' (read) mode and loads the JSON data into a Python dictionary, which is then returned.

    If the file does not exist, indicating that the route has not been cached previously, the function returns None.
    This design allows the calling code to decide how to handle the absence of cached data, such as fetching fresh data
    if necessary.

    """
    file_path = cache_file_path(origin, destination, departure_time)
    if Path.exists(file_path):
        with Path.open(file_path) as file:
            return json.load(file)
    return None


def fetch_all_routes(apikey: str, origins: List[Location], destinations: List[Location],
                     traffic: str, departure_times):
    """Fetch route data for multiple origin-destination pairs concurrently, using cache.

    Parameters
    ----------
    apikey : str
        The API key for accessing the Google Routes API.
    origins : list of Location
        A list of Location objects representing the origins.
    destinations : list of Location
        A list of Location objects representing the destinations.
    traffic : str
        Traffic awareness mode ('TRAFFIC_UNAWARE', 'TRAFFIC_AWARE', 'TRAFFIC_AWARE_OPTIMAL').
    departure_times : list of tuples
        A list of tuples representing the departure times for each route.

    Returns
    -------
    list
        A list of route data, each corresponding to the pair of origin and destination.

    Notes
    -----
    The function employs a ThreadPoolExecutor for concurrent execution, improving performance for multiple route calculations.
    It first checks if the route data is available in the cache. If so, it retrieves the data directly from the cache.
    For routes not available in the cache, it calculates them using the 'calculateRoute' function in a parallel manner.
    Calculated routes are then cached for future use.
    The results are sorted and returned in the order corresponding to the input lists of origins and destinations.
    Exceptions during route calculations are caught and printed, but do not halt the execution of the entire function.

    """
    results_with_keys = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_route = {}
        for i, (origin, destination, dep_time) in enumerate(
                zip(origins, destinations, departure_times, strict=False), ):
            # Find the data in the cache
            cached_data = load_from_cache(origin, destination, dep_time)
            if cached_data:
                # If data is found in the cache, add it to the results
                results_with_keys.append((i, cached_data))
            else:
                # If data is not in the cache, start the task
                future = executor.submit(calculate_route, apikey, origin, destination, traffic,
                    dep_time, )
                future_to_route[future] = (i, origin, destination, dep_time)

        for future in concurrent.futures.as_completed(future_to_route):
            index, origin, destination, dep_time = future_to_route[future]
            try:
                data = future.result()
                save_to_cache(data, origin, destination, dep_time)
                results_with_keys.append((index, data))
            except Exception as exc:
                print(f"Exception for request from {origin} to {destination} at {dep_time}: {exc}")

    # Sort the results based on the index key
    sorted_results = [data for _, data in sorted(results_with_keys, key=lambda x: x[0])]
    return sorted_results
