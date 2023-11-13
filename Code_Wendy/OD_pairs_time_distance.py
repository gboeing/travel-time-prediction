import multiprocessing as mp
import time

import osmnx as ox
import pandas as pd

from Code_Jessica.functions import get_turn_penalty_dict, shortest_path_turn_penalty, add_edge_traffic_times


def calculate(G, source, target, turn_penalty):
    """
    Calculate trip distance and travel time of an origin and destination in a graph based on
    3 kinds of routing algorithm: traffic signal and turn penalty, shortest freeflow travel time,
    and shortest path.

      Parameters
      ----------
      G : networkx.MultiGraph
          An undirected, unprojected graph with 'bearing' attributes on each edge.
      source : int
          The osmid of the origin node
      target : int
          the osmid of the destination node
      turning_circle_time : int, optional
          The time penalty for navigating a turning circle or roundabout (default: 5).
      crossing_time : int, optional
          The time penalty for crossing a pedestrian crossing (default: 5).
      turn_penalty : dict
          turn penalty dictionary

      Returns
      -------
      A tuple of origin's osmid, destination's osmid, turn and traffic signal penalized total travel time,
      shortest freeflow travel time, shortest path travel time, turn and traffic signal penalized travel distance,
      the distance of freeflow travel time routing , shortest path distance
      """
    try:
        penalty_route = shortest_path_turn_penalty(G, source, target, weight='total_time', penalty=turn_penalty)
        penalized_total_time = sum(ox.utils_graph.get_route_edge_attributes(G, penalty_route, 'total_time'))
        penalized_distance = sum(ox.utils_graph.get_route_edge_attributes(G, penalty_route, 'length'))
    except:
        # If the path is unsolvable, return -1, -1
        penalized_total_time, penalized_distance = -1, -1

    try:
        freeflow_route = shortest_path_turn_penalty(G, source, target, weight='travel_time', penalty={})
        freeflow_travel_time = sum(ox.utils_graph.get_route_edge_attributes(G, freeflow_route, 'travel_time'))
        freeflow_distance = sum(ox.utils_graph.get_route_edge_attributes(G, freeflow_route, 'length'))
    except:
        # If the path is unsolvable, return -1, -1
        freeflow_travel_time, freeflow_distance = -1, -1

    try:
        distance_route = shortest_path_turn_penalty(G, source, target, weight='length', penalty={})
        shortest_path_travel_time = sum(ox.utils_graph.get_route_edge_attributes(G, distance_route, 'travel_time'))
        shortest_path_distance = sum(ox.utils_graph.get_route_edge_attributes(G, distance_route, 'length'))
    except:
        # If the path is unsolvable, return -1, -1
        shortest_path_travel_time, shortest_path_distance = -1, -1

    return (source, target, penalized_total_time, freeflow_travel_time, shortest_path_travel_time, penalized_distance,
            freeflow_distance, shortest_path_distance)


def run():
    cpus = mp.cpu_count() - 2

    file_path = '../Data/'
    output_file_path = file_path + 'Output/'
    traffic_time_config = {
        'traffic_signals_time': 2,
        'stop_time': 2,
        'turning_circle_time': 0,
        'crossing_time': 1.5,
        'give_way_time': 1.5,
        'mini_roundabout_time': 1.5}

    uber_1m = pd.read_csv(output_file_path + 'OD_pairs_uber_1m_strongly.csv')
    # a subset of the sampled OD pairs, as we spread ran it across machines
    uber_1m_sample = uber_1m[100000:200000]

    G = ox.io.load_graphml(output_file_path + 'LA_clip_convex_strong_network.graphml')
    turn_penalty = get_turn_penalty_dict(G)

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.bearing.add_edge_bearings(G)
    G = add_edge_traffic_times(G, **traffic_time_config)

    args = (
        (G, uber_1m_sample.iloc[i]['oid'], uber_1m_sample.iloc[i]['did'], turn_penalty)
        for i in range(len(uber_1m_sample))
    )

    cur = time.time()
    pool = mp.Pool(cpus)
    res = pool.starmap_async(calculate, args)
    all_results = res.get()

    pool.close()
    pool.join()
    print(time.time() - cur)

    df = pd.DataFrame(all_results, columns=['oid', 'did', 'penalized_total_time', 'freeflow_travel_time',
                                            'shortest_path_travel_time', 'penalized_distance',
                                            'freeflow_distance', 'shortest_path_distance'])
    df.to_csv(output_file_path + "distance_time_result_100000_200000.csv")


if __name__ == '__main__':
    run()
