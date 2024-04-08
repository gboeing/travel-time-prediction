# import packages
import multiprocessing as mp
import time
import osmnx as ox
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from functions.penalty_functions import get_turn_penalty_dict, shortest_path_turn_penalty, add_edge_traffic_times


def calculate(G, gdf_nodes, source, target, turn_penalty, sy, sx, ty, tx, hod, uber_time, algorithm='penalized',
              left_turn_penalty=6.2, right_turn_penalty=8.2, u_turn_penalty=13.1):
    """
    Calculate trip distance and travel time of an origin and destination in a graph based on
    3 kinds of routing algorithm: traffic signal and turn penalty, shortest freeflow travel time,
    and shortest path.

      Parameters
      ----------
      G : networkx.MultiGraph
          An undirected, unprojected graph with 'bearing' attributes on each edge.
      gdf_nodes: geopandas.GeoDataFrame
          The nodes of graph G in Geodataframe format
      source : int
          The osmid of the origin node
      target : int
          the osmid of the destination node
      turn_penalty : dict
          dictionary mapping tuples (u, v, m) to their corresponding turn penalties, generated from
          get_turn_penalty_dict() function
      sy : float
          the latitude of the origin node
      sx : float
          the longitude of the origin node
      ty : float
          the latitude of the destination node
      tx : float
          the longitude of the destination node
      hod : int
          The hour of day of the set departure time (of uber movement data)
      uber_time : float
          The travel time in seconds based on uber movement
      algorithm : str
          Choosing the algorithm for routing: 'penalized' (with traffic controls and turn penalties), 'freeflow' (shortest
          time routing without considering any traffic controls and turn penalties), 'shortest_distance' (shortest
          distance routing)
      left_turn_penalty : float (default: 5.7s)
          Penalty in seconds for left turns, set to 0
      right_turn_penalty : float (default: 7.7s)
          Penalty in seconds for right turns
      u_turn_penalty : float (default: 11.1s)
          Penalty in seconds for u turns

      Returns
      -------
      A tuple of origin's osmid, destination's osmid, the latitude of the origin node， the longitude of the origin node,
      the latitude of the destination node,the longitude of the destination node, travel time without considering
      turn penalties, total travel time considering turn penalties, total distance, route (a list of osmid),
      count of traffic signals, count of stop sings, count of crossing, count of give way signs,
      count of mini roundabout, count of left turns, count of right turns, count of u turns.
      """
    try:
        if algorithm == 'penalized':
            route = shortest_path_turn_penalty(G, source, target, weight='total_time', penalty=turn_penalty)
        elif algorithm == 'freeflow':
            route = shortest_path_turn_penalty(G, source, target, weight='travel_time', penalty={})
        elif algorithm == 'shortest_distance':
            route = shortest_path_turn_penalty(G, source, target, weight='length', penalty={})

        route_edges = ox.utils_graph.route_to_gdf(G, route)
        startid = route_edges.index.get_level_values('u').to_list()
        signal_count = len(
            gdf_nodes[gdf_nodes.index.isin(startid)][gdf_nodes["highway"] == "traffic_signals"])
        stop_count = len(gdf_nodes[gdf_nodes.index.isin(startid)][gdf_nodes["highway"] == "stop"])
        crossing_count = len(
            gdf_nodes[gdf_nodes.index.isin(startid)][gdf_nodes["highway"] == "crossing"])
        give_way_count = len(
            gdf_nodes[gdf_nodes.index.isin(startid)][gdf_nodes["highway"] == "give_way"])
        mini_roundabout_count = len(
            gdf_nodes[gdf_nodes.index.isin(startid)][gdf_nodes["highway"] == "mini_roundabout"])

        route_index = route_edges.reset_index()
        route_index["turn_degree"] = 0
        for a in range(1, len(route_index)):
            prev_bearing = route_index.loc[a - 1, "bearing"]
            curr_bearing = route_index.loc[a, "bearing"]
            turn_degree = curr_bearing - prev_bearing
            route_index.loc[a, "turn_degree"] = turn_degree % 360
        route_index["turn_type"] = "straight"
        for b in range(1, len(route_index)):
            if 0 <= route_index.loc[b, "turn_degree"]:
                if route_index.loc[b, "turn_degree"] <= 45:
                    route_index.loc[b, "turn_type"] = "straight"
                if 45 < route_index.loc[b, "turn_degree"] <= 135:
                    route_index.loc[b, "turn_type"] = "right_turn"
                if 135 < route_index.loc[b, "turn_degree"] <= 225:
                    route_index.loc[b, "turn_type"] = "u_turn"
                if 225 < route_index.loc[b, "turn_degree"] <= 315:
                    route_index.loc[b, "turn_type"] = "left_turn"
                if 315 < route_index.loc[b, "turn_degree"] <= 360:
                    route_index.loc[b, "turn_type"] = "straight"
        left_count = len(route_index[route_index["turn_type"] == "left_turn"])
        right_count = len(route_index[route_index["turn_type"] == "right_turn"])
        u_count = len(route_index[route_index["turn_type"] == "u_turn"])


        travel_time = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'travel_time'))
        total_time = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'total_time'))
        total_time_with_turn_penalty = total_time + left_turn_penalty * left_count + right_turn_penalty * right_count + u_turn_penalty * u_count
        distance = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))

    except:
        # If the path is unsolvable, return -1, -1
        travel_time, distance = -1, -1
    return (source, target, sy, sx, ty, tx, hod, uber_time, travel_time, total_time, total_time_with_turn_penalty,
            distance, route, signal_count,
            stop_count, crossing_count, give_way_count, mini_roundabout_count,
            left_count, right_count, u_count)


def run():
    cpus = mp.cpu_count() - 2

    file_path = '../Data/'
    output_file_path = file_path + 'Output/'
    traffic_time_config = {
        'traffic_signals_time': 3.8,
        'stop_time': 5.3,
        'turning_circle_time': 0,
        'crossing_time': 0,
        'give_way_time': 0,
        'mini_roundabout_time': 0}

    all_OD_pairs = pd.read_csv(output_file_path + 'OD3am_20000.csv')
    # a subset of the sampled OD pairs, as we may spread ran it across machines
    OD_pair_sample = all_OD_pairs

    G = ox.io.load_graphml(output_file_path + 'LA_clip_convex_strong_network.graphml')

    # set up turn penalty dictionary

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.bearing.add_edge_bearings(G)
    G = add_edge_traffic_times(G, **traffic_time_config)
    turn_penalty = get_turn_penalty_dict(G, left_turn_penalty=6.2, right_turn_penalty=8.2, u_turn_penalty=13.1)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # choose to calculate the penalized routing
    routing_algorithm = 'penalized'
    args = (
        (G, gdf_nodes, OD_pair_sample.iloc[i]['oid'], OD_pair_sample.iloc[i]['did'], turn_penalty,
         OD_pair_sample.iloc[i]['oy'], OD_pair_sample.iloc[i]['ox'],
         OD_pair_sample.iloc[i]['dy'], OD_pair_sample.iloc[i]['dx'],
         OD_pair_sample.iloc[i]['hod'], OD_pair_sample.iloc[i]['mean_travel_time'], routing_algorithm ,6.2, 8.2, 13.1)
        for i in range(len(OD_pair_sample))
    )

    cur = time.time()
    pool = mp.Pool(cpus)
    res = pool.starmap_async(calculate, args)
    all_results = res.get()

    pool.close()
    pool.join()
    print(time.time() - cur)

    df = pd.DataFrame(all_results, columns=['oid', 'did', 'oy', 'ox', 'dy', 'dx', 'hour_of_day', 'uber_time',
                                            'travel_time', 'total_time','total_time_with_turn_penalty', 'distance',
                                            'route',
                                            'signal_count',
                                            'stop_count', 'crossing_count',
                                            'give_way_count',
                                            'mini_roundabout_count',
                                            'left_count', 'right_count', 'u_count'])

    df.to_csv(output_file_path + 'result0121/' + f"{routing_algorithm}_OD3am_20000_model1.csv")


if __name__ == '__main__':
    run()
