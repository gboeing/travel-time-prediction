class Location:
    """
    This class stores the longitude (lon), latitude (lat), and address (addr) of a geographic location.

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




import re
def parseDataframeToLocations(dataframe):
    """
    Convert a DataFrame containing geometry and address information into a list of Location objects.

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
        point = dataframe.geometry.iloc[i]
        # define a regular expression pattern
        pattern = r'\((.*?)\)'
        # use the regular expression to find all matches in the point
        longLat = re.findall(pattern, point)
        # find the index of the first space character 
        longLat = longLat[0]
        splitIndex = longLat.find(" ")
        longitude = longLat[0:splitIndex]
        latitude = longLat[splitIndex + 1:len(longLat)]

        locations.append(Location(longitude, latitude, dataframe.address.iloc[i]))
    
    return locations




import requests
from datetime import datetime
import pytz
def calculateRoute(apikey, orig: Location, dest: Location, traffic='TRAFFIC_UNAWARE', departure_time=None):
    """
    Calculate a driving route between two locations with routingPreference(traffic) and departure_time.

    This function calculates a driving route between two locations (origin and destination) using the Google Routes API.
    It considers traffic and allows to specify the departure time.

    Parameters
    ----------
    apikey : str
        The Google Routes API key for authentication.
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
    route_data : dict
        A dictionary containing route information, including duration, distance, and encoded polyline.

    Notes
    -----
    - The function converts the departure_time tuple to a datetime object and adjusts it to the US/Pacific timezone.
    - It sends a request to the Google Routes API with the specified parameters.
    - The API key is used for authentication, and the departure time is included in the request.
    - The returned route_data dictionary contains details about the calculated route.
    """
    # convert the departure_time tuple to a datetime object in the US/Pacific timezone
    local_tz = pytz.timezone('US/Pacific')
    departure_time_str = None  # default value for departure_time_str

    if departure_time is not None:
        departure_time = datetime(*departure_time, tzinfo=local_tz)
        departure_time_utc = departure_time.astimezone(pytz.utc)
        departure_time_str = departure_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # setup the headers and body for the request
    heads = {
        "X-Goog-Api-Key": apikey,
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
    }

    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": orig.lat,
                    "longitude": orig.lon
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": dest.lat,
                    "longitude": dest.lon
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": traffic
    }

    # add departureTime to the body if it is not None
    if departure_time_str is not None:
        body["departureTime"] = departure_time_str

    # send the request to the Google Routes API
    response = requests.post("https://routes.googleapis.com/directions/v2:computeRoutes", headers=heads, json=body)
    return response.json()




import pandas as pd
def createDataFrame(apikey, orig, dest, traffic='TRAFFIC_UNAWARE', departure_time=None):
    """
    Create a DataFrame with route information between two locations.

    This function calculates driving routes between two locations (origin and destination) using the Google Routes API.

    Parameters
    ----------
    apikey : str
        The Google Routes API key for authentication.
    orig : Location
        Class object representing the origin of the route.
    dest : Location
        Class object representing the destination of the route.
    traffic : str, optional
        Traffic awareness mode ('TRAFFIC_UNAWARE', 'TRAFFIC_AWARE', or 'TRAFFIC_AWARE_OPTIMAL').
    departure_time : tuple, optional
        A tuple representing the departure time in the format (year, month, day, hour, minute) (default: None).

    Returns
    -------
    routes_df : pandas.DataFrame
        A Pandas DataFrame containing route information, including origin, destination, distance, duration, and polyline.

    Notes
    -----
    - The function calls the `calculateRoute` function to obtain driving route information.
    - It extracts distance, duration, and polyline data from the API response.
    - The route data is organized into a Pandas DataFrame.
    - The DataFrame includes columns for origin, destination, distance (in meters), duration (in seconds), and polyline.
    """
    # call the calculateRoute function for a pair of origin and destination
    response = calculateRoute(apikey, orig, dest, traffic, departure_time)
    
    route_data = []
    
    # check if response contains 'routes' and if the first route is available
    if response is not None and 'routes' in response and len(response['routes']) > 0:
        # extract the first route
        route = response['routes'][0]
        distance = route['distanceMeters']
        duration = int(route['duration'].replace('s', ''))
        polyline = route['polyline']['encodedPolyline']
        
        # append the extracted data to the list
        route_data.append({
            'origin': f"({orig.lat}, {orig.lon})",
            'destination': f"({dest.lat}, {dest.lon})",
            'distance': distance,
            'duration': duration,
            'polyline': polyline
        })
    else:
        # append None values if no routes are found
        route_data.append({
            'origin': f"({orig.lat}, {orig.lon})",
            'destination': f"({dest.lat}, {dest.lon})",
            'distance': None,
            'duration': None,
            'polyline': None
        })
    
    # convert the list of dictionaries to a DataFrame
    routes_df = pd.DataFrame(route_data)
    
    return routes_df




import os
import json
cache_folder = 'route_cache'
os.makedirs(cache_folder, exist_ok=True)

def cache_file_path(origin, destination):
    """
    Generate a file path for caching route data based on the origin and destination coordinates.

    Parameters
    ----------
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.

    Returns
    -------
    str
        A string representing the path to the cache file. This path includes the filename which is constructed
        based on the latitude and longitude of the origin and destination.

    Notes
    -----
    This function is used to create a standardized file path for caching route information.
    The filename is formed using the latitude and longitude of the origin and destination points.
    The file is saved with a '.json' extension and stored in a predefined cache folder.
    """
    filename = f"route_{origin.lat}_{origin.lon}_to_{destination.lat}_{destination.lon}.json"
    return os.path.join(cache_folder, filename)




def save_to_cache(data, origin, destination):
    """
    Save route data to a cache file.

    Parameters
    ----------
    data : dict
        The data to be cached. This would be route information in a dictionary format.
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.

    Returns
    -------
    None
        This function does not return a value. It performs the operation of writing data to a file.

    Notes
    -----
    This function saves route data to a cache file to avoid redundant computations or API calls in the future.
    """
    file_path = cache_file_path(origin, destination)
    with open(file_path, 'w') as file:
        json.dump(data, file)




def load_from_cache(origin, destination):
    """
    Load route data from a cache file, if it exists.

    Parameters
    ----------
    origin : Location
        A Location object representing the starting point of the route, with latitude and longitude attributes.
    destination : Location
        A Location object representing the destination point of the route, with latitude and longitude attributes.

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
    file_path = cache_file_path(origin, destination)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return None


import concurrent.futures
def fetch_all_routes(apikey, origins, destinations, traffic='TRAFFIC_UNAWARE', departure_time=None):
    """
    Fetch route data for multiple origin-destination pairs, using caching and multithreading.

    Parameters
    ----------
    apikey : str
        The API key used to authenticate requests to the routing service.
    origins : list of Location
        A list of Location objects representing the starting points for the routes.
    destinations : list of Location
        A list of Location objects representing the destination points for the routes.
    traffic : str, optional
        The mode of traffic awareness. Default is 'TRAFFIC_UNAWARE'.
    departure_time : datetime, optional
        The departure time for the routes. If not provided, current time is assumed.

    Returns
    -------
    list of dict
        A list of route data, each represented as a dictionary.

    Notes
    -----
    This function uses multithreading to concurrently fetch route data for multiple origin-destination pairs.
    It first checks if the route data is available in the cache. If so, it adds the cached data to the results.
    If not, it submits a task to calculate the route to a ThreadPoolExecutor.

    The function uses a maximum of 10 worker threads to handle the route calculations. Each route calculation
    is submitted as a separate task to the executor.

    After submitting all tasks, the function waits for each task to complete. Completed tasks have their results
    checked: if successful, the data is cached and added to the results; if an exception occurs, it is printed.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_route = {}
        for origin, destination in zip(origins, destinations):
            cached_data = load_from_cache(origin, destination)
            if cached_data:
                results.append(cached_data)
            else:
                future = executor.submit(calculateRoute, apikey, origin, destination, traffic, departure_time)
                future_to_route[future] = (origin, destination)

        for future in concurrent.futures.as_completed(future_to_route):
            origin, destination = future_to_route[future]
            try:
                data = future.result()
                save_to_cache(data, origin, destination)  
                results.append(data)
            except Exception as exc:
                print(f'{origin} to {destination} generated an exception: {exc}')
    return results