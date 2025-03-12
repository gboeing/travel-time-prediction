"""Calculating the travel time and distance of OD in a graph based on routing algorithms."""

import multiprocessing as mp
import time
import warnings
from heapq import heappop, heappush
from itertools import count

import networkx as nx
import osmnx as ox
import pandas as pd

import constants

warnings.simplefilter(action="ignore", category=FutureWarning)

_LEFT_TURN_DEGREE_LOWER_BOUND = 225
_LEFT_TURN_DEGREE_UPPER_BOUND = 315
_SLIGHT_LEFT_TURN_DEGREE_LOWER_BOUND = 315
_SLIGHT_LEFT_TURN_DEGREE_UPPER_BOUND = 330
_RIGHT_TURN_DEGREE_LOWER_BOUND = 45
_RIGHT_TURN_DEGREE_UPPER_BOUND = 135
_SLIGHT_RIGHT_TURN_DEGREE_LOWER_BOUND = 30
_SLIGHT_RIGHT_TURN_DEGREE_UPPER_BOUND = 45
_U_TURN_DEGREE_LOWER_BOUND = 135
_U_TURN_DEGREE_UPPER_BOUND = 225


def add_edge_control_non_simplified(
    graph: nx.MultiGraph,
    traffic_signals_time: float = 2,
    stop_time: float = 2,
    turning_circle_time: float = 0,
    crossing_time: float = 1.5,
    give_way_time: float = 1.5,
    mini_roundabout_time: float = 1.5,
) -> nx.MultiGraph:
    """Calculate traffic time penalties for different features of edge attributes in a graph.

    This function assigns traffic time penalties to edges.
    It considers various highway types, such as traffic signals, stop signs, turning circles,
    crossings, give-way signs, and mini roundabouts, and assigns corresponding time penalties to
    these features.

    Parameters
    ----------
    graph : networkx.MultiGraph
      An undirected, unprojected graph with 'bearing' attributes on each edge.
    traffic_signals_time : float, optional
      The time penalty for passing through a traffic signal-controlled intersection (default: 2).
    stop_time : float, optional
      The time penalty for stopping at a stop sign or stop-controlled intersection (default: 2).
    turning_circle_time : float, optional
      The time penalty for navigating a turning circle or roundabout (default: 0).
    crossing_time : float, optional
      The time penalty for crossing a pedestrian crossing (default: 1.5).
    give_way_time : float, optional
      The time penalty for yielding at a give-way or yield sign (default: 1.5).
    mini_roundabout_time : float, optional
      The time penalty for navigating a mini roundabout (default: 1.5).

    Returns
    -------
    G : networkx.MultiGraph
      The graph with updated 'traffic_time' and 'total_time' attributes on each edge, representing
      the calculated traffic-related time penalties.

    """
    for u, v, key, data in graph.edges(data=True, keys=True):
        # check if 'highway' tag is in destination node
        if v in graph.nodes and "highway" in graph.nodes[v]:
            highway_value = graph.nodes[v]["highway"]
            # map the time values of traffic penalties
            if highway_value == "traffic_signals":
                if "traffic_signals:direction" in graph.nodes[v]:
                    if (
                        graph.nodes[v]["traffic_signals:direction"] == "forward"
                        and graph[u][v][key]["reversed"]
                    ) or (
                        graph.nodes[v]["traffic_signals:direction"] == "backward"
                        and not graph[u][v][key]["reversed"]
                    ):
                        traffic_time = 0
                        traffic_control = ""
                    else:
                        traffic_time = traffic_signals_time
                        traffic_control = "traffic_signals"
                elif "direction" in graph.nodes[v]:
                    if (
                        graph.nodes[v]["direction"] == "forward" and graph[u][v][key]["reversed"]
                    ) or (
                        graph.nodes[v]["direction"] == "backward"
                        and not graph[u][v][key]["reversed"]
                    ):
                        traffic_time = 0
                        traffic_control = ""
                    else:
                        traffic_time = traffic_signals_time
                        traffic_control = "traffic_signals"
                else:
                    traffic_time = traffic_signals_time
                    traffic_control = "traffic_signals"
            elif highway_value == "stop":
                if "stop:direction" in graph.nodes[v]:
                    if (
                        graph.nodes[v]["stop:direction"] == "forward"
                        and graph[u][v][key]["reversed"]
                    ) or (
                        graph.nodes[v]["stop:direction"] == "backward"
                        and not graph[u][v][key]["reversed"]
                    ):
                        traffic_time = 0
                        traffic_control = ""
                    else:
                        traffic_time = stop_time
                        traffic_control = "stop"
                elif "direction" in graph.nodes[v]:
                    if (
                        graph.nodes[v]["direction"] == "forward" and graph[u][v][key]["reversed"]
                    ) or (
                        graph.nodes[v]["direction"] == "backward"
                        and not graph[u][v][key]["reversed"]
                    ):
                        traffic_time = 0
                        traffic_control = ""
                    else:
                        traffic_time = stop_time
                        traffic_control = "stop"
                else:
                    traffic_time = stop_time
                    traffic_control = "stop"
            elif highway_value == "turning_circle":
                traffic_time = turning_circle_time
                traffic_control = "turning_circle"
            elif highway_value == "crossing":
                traffic_time = crossing_time
                traffic_control = "crossing"
            elif highway_value == "give_way":
                if "direction" in graph.nodes[v]:
                    if (
                        graph.nodes[v]["direction"] == "forward" and graph[u][v][key]["reversed"]
                    ) or (
                        graph.nodes[v]["direction"] == "backward"
                        and not graph[u][v][key]["reversed"]
                    ):
                        traffic_time = 0
                        traffic_control = ""
                    else:
                        traffic_time = give_way_time
                        traffic_control = "give_way"
                else:
                    traffic_time = give_way_time
                    traffic_control = "give_way"

            elif highway_value == "mini_roundabout":
                traffic_time = mini_roundabout_time
                traffic_control = "mini_roundabout"
            else:
                traffic_time = 0
                traffic_control = ""
        else:
            traffic_time = 0
            traffic_control = ""

        graph[u][v][key]["traffic_time"] = traffic_time
        graph[u][v][key]["traffic_control"] = traffic_control
        # calculate 'total_time' by adding 'travel_time' and 'traffic_time'
        # add 'total_time' attribute to the edge
        graph[u][v][key]["total_time"] = data.get("travel_time") + traffic_time
    return graph


def get_slight_turn_penalty_dict(
    graph: nx.MultiGraph,
    left_turn_penalty: float = 30,
    slight_left_turn_penalty: float = 20,
    right_turn_penalty: float = 10,
    slight_right_turn_penalty: float = 20,
    u_turn_penalty: float = 90,
) -> dict[tuple:float]:
    """Calculate turn penalties for different types of turns at intersections in a graph.

    This function computes turn penalties for various types of turns (left, right, U-turn) at
    intersections within a road network represented as a graph. The penalties are based on the
    difference in bearing between the edges that meet at each intersection.

    Parameters
    ----------
    graph : networkx.MultiGraph
      A directed graph representing a road network with 'bearing' attributes on each edge.
    left_turn_penalty : float, optional
      The penalty for making a left turn at an intersection (default: 30).
    slight_left_turn_penalty : float, optional
      The penalty for making a slight left turn at an intersection (default: 20).
    right_turn_penalty : float, optional
      The penalty for making a right turn at an intersection (default: 10).
    slight_right_turn_penalty : float, optional
      The penalty for making a slight right turn at an intersection (default: 20).
    u_turn_penalty : float, optional
      The penalty for making a U-turn at an intersection (default: 90).

    Returns
    -------
    penalty : dict
      A dictionary mapping tuples (u, v, m) to their corresponding turn penalties, where:
      - (u, v) represents the incoming road segment,
      - (v, m) represents the outgoing road segment, and
      - The associated value represents the calculated turn penalty based on the difference in
      bearing.

    """
    penalty = {}
    # iterate all nodes in G
    for v, _ in graph.nodes(data=True):
        # for each previous node 'u' of node 'v'

        for u, edge_keys in graph.pred[v].items():
            # for each edge from 'u' to 'v'
            for key in edge_keys:
                # for each next node 'm' of node 'v'
                for m, next_edge_keys in graph[v].items():
                    # for each edge from 'v' to 'm'
                    for next_key in next_edge_keys:
                        # check both edges (u-v, v-m)
                        if "bearing" in graph[u][v][key] and "bearing" in graph[v][m][next_key]:
                            # calculate the difference between the bearings of the two edges
                            bearing_diff = (
                                graph[v][m][next_key]["bearing"] - graph[u][v][key]["bearing"]
                            ) % 360
                            # add penalties based on the difference
                            if (
                                _LEFT_TURN_DEGREE_LOWER_BOUND
                                < bearing_diff
                                <= _LEFT_TURN_DEGREE_UPPER_BOUND
                            ):
                                penalty[(u, v, m)] = left_turn_penalty  # left turn
                            elif (
                                _SLIGHT_LEFT_TURN_DEGREE_LOWER_BOUND
                                < bearing_diff
                                <= _SLIGHT_LEFT_TURN_DEGREE_UPPER_BOUND
                            ):
                                penalty[(u, v, m)] = slight_left_turn_penalty  # light left turn
                            elif (
                                _RIGHT_TURN_DEGREE_LOWER_BOUND
                                < bearing_diff
                                <= _RIGHT_TURN_DEGREE_UPPER_BOUND
                            ):
                                penalty[(u, v, m)] = right_turn_penalty  # right turn
                            elif (
                                _SLIGHT_RIGHT_TURN_DEGREE_LOWER_BOUND
                                <= bearing_diff
                                <= _SLIGHT_RIGHT_TURN_DEGREE_UPPER_BOUND
                            ):
                                penalty[(u, v, m)] = slight_right_turn_penalty  # light right turn
                            elif (
                                _U_TURN_DEGREE_LOWER_BOUND
                                < bearing_diff
                                <= _U_TURN_DEGREE_UPPER_BOUND
                            ):
                                penalty[(u, v, m)] = u_turn_penalty  # U turn
                            else:
                                penalty[(u, v, m)] = 0  # Straight
    return penalty


# Source: https://github.com/maxtmng/shortest_path_turn_penalty/
## used the function without any modification
def shortest_path_turn_penalty(
    graph: nx.MultiGraph,
    source: int,
    target: int,
    weight: str,
    penalty: dict | None = None,
    next_node: int | None = None,
) -> list[int]:
    """Use Dijkstra's algorithm to find the shortest weighted paths with turn penalty.

    This function is adapted from networkx.algorithms.shortest_paths.weighted._dijkstra_multisource.
    The turn penalty implementation is based on:
    Ziliaskopoulos, A.K., Mahmassani, H.S., 1996. A note on least time path computation considering
    delays and prohibitions for intersection movements. Transportation Research Part B:
    Methodological 30, 359-367. https://doi.org/10.1016/0191-2615(96)00001-X

    Parameters
    ----------
    graph : NetworkX graph
        The graph of the street network.
    source : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.
    target : node label, single node or a list
        Ending node (or a list of ending nodes) for path. Search is halted when any target is found.
    weight: str
        The name of the edge attribute that represents the weight of an edge,
        or None to indicate a hidden edge
    penalty : dict, optional (default=None)
        Dictionary containing turn penalties. The key is a tuple (u, v, m) where
        u, v are the nodes of the current edge and m is the next node.
    next_node : node, optional (default=None)
        Next node to consider from the source.

    Returns
    -------
    list of nodes
        Path from source to target.

    Raises
    ------
    NodeNotFound
        If the source or target is not in `G`.
    ValueError
        If contradictory paths are found due to negative weights.

    """
    if penalty is None:
        penalty = {}

    succ_graph = graph.adj  # For speed-up (and works for both directed and undirected graphs)
    weight = nx.algorithms.shortest_paths.weighted._weight_function(graph, weight)
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    paths = {source: [source]}
    target_list = [target] if not isinstance(target, list) else target
    reached_target = None
    seen = {}
    c = count()
    fringe = []
    seen[source] = {}
    if next_node is None:
        for m in succ_graph[source]:
            seen[source][m] = 0
            push(fringe, (0, next(c), source, m))
    else:
        push(fringe, (0, next(c), source, next_node))
    while fringe:
        (d, _, v, m) = pop(fringe)
        u = m
        if v in dist:
            if u in dist[v]:
                continue  # already searched this node.
        else:
            dist[v] = {}
        dist[v][u] = d
        if v in target_list:
            reached_target = v
            break
        e = graph[v][u]
        for m in succ_graph[u]:
            cost = weight(v, u, e)
            if (v, u, m) in penalty:
                cost += penalty[v, u, m]

            if cost is None:
                continue
            vu_dist = dist[v][u] + cost
            if u in dist:
                if m in dist[u]:
                    u_dist = dist[u][m]
                    if vu_dist < u_dist:
                        error_message = "Contradictory paths found: negative weights"
                        raise ValueError(error_message)
            elif u not in seen or m not in seen[u] or vu_dist < seen[u][m]:
                if u not in seen:
                    seen[u] = {}
                seen[u][m] = vu_dist
                push(fringe, (vu_dist, next(c), u, m))
                if paths is not None:
                    paths[u] = paths[v] + [u]
    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return paths[reached_target]


def calculate(
    graph: nx.MultiGraph,
    source: int,
    target: int,
    turn_penalty: dict,
    sy: float,
    sx: float,
    ty: float,
    tx: float,
    hod: int,
    uber_time: float,
    algorithm: str = "freeflow",
    left_turn_penalty: float = 30,
    slight_left_turn_penalty: float = 10,
    right_turn_penalty: float = 15,
    slight_right_turn_penalty: float = 5,
    u_turn_penalty: float = 50,
) -> tuple:
    """Calculate trip distance and travel time of an origin and destination.

    Parameters
    ----------
    graph : networkx.MultiGraph
      An undirected, unprojected graph with 'bearing' attributes on each edge.
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
      Algorithm for routing: 'penalized' (with traffic controls and turn penalties), 'freeflow'
      (shortest time routing without considering any traffic controls and turn penalties),
      'shortest_distance' (shortest distance routing)
    left_turn_penalty : float (default: 30s)
      Penalty in seconds for left turns (225-315 degrees), set to 30
    slight_left_turn_penalty : float (default: 10s)
      Penalty in seconds for slight left turns (315 - 330 degrees), set to 10
    right_turn_penalty : float (default: 15s)
      Penalty in seconds for right turns (45-135 degrees), set to 15
    slight_right_turn_penalty : float (default: 5s)
      Penalty in seconds for slight right turns (30-45 degrees), set to 5
    u_turn_penalty : float (default: 50s)
      Penalty in seconds for u turns (135-225 degrees), set for 50s

    Returns
    -------
      A tuple of origin's osmid, destination's osmid, the latitude of the origin node, the
      longitude of the origin node, the latitude of the destination node,the longitude of the
      destination node, travel time without considering turn penalties, total travel time
      considering traffic control penalties and turn penalties (if any), total distance, route (a
      list of osmid), count of traffic signals, count of stop sings, count of crossing, count of
      give way signs, count of mini roundabout, count of left turns, count of right turns, count of
      u turns.

    """
    try:
        if algorithm == "penalized":
            route = shortest_path_turn_penalty(
                graph,
                source,
                target,
                weight="total_time",
                penalty=turn_penalty,
            )
        elif algorithm == "freeflow":
            route = shortest_path_turn_penalty(
                graph,
                source,
                target,
                weight="travel_time",
                penalty=None,
            )
        elif algorithm == "shortest_distance":
            route = shortest_path_turn_penalty(graph, source, target, weight="length", penalty=None)

        route_edges = ox.utils_graph.route_to_gdf(graph, route)
        signal_count = len(route_edges[route_edges["traffic_control"] == "traffic_signals"])
        stop_count = len(route_edges[route_edges["traffic_control"] == "stop"])
        crossing_count = len(route_edges[route_edges["traffic_control"] == "crossing"])
        give_way_count = len(route_edges[route_edges["traffic_control"] == "give_way"])
        mini_roundabout_count = len(
            route_edges[route_edges["traffic_control"] == "mini_roundabout"],
        )

        route_index = route_edges.reset_index()
        route_index["turn_degree"] = 0
        for a in range(1, len(route_index)):
            prev_bearing = route_index.loc[a - 1, "bearing"]
            curr_bearing = route_index.loc[a, "bearing"]
            turn_degree = curr_bearing - prev_bearing
            route_index.loc[a, "turn_degree"] = turn_degree % 360
        route_index["turn_type"] = "straight"
        for b in range(1, len(route_index)):
            if (
                _LEFT_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _LEFT_TURN_DEGREE_UPPER_BOUND
            ):
                route_index.loc[b, "turn_type"] = "left_turn"
            if (
                _SLIGHT_LEFT_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _SLIGHT_LEFT_TURN_DEGREE_UPPER_BOUND
            ):
                route_index.loc[b, "turn_type"] = "slight_left_turn"
            if (
                _RIGHT_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _RIGHT_TURN_DEGREE_UPPER_BOUND
            ):
                route_index.loc[b, "turn_type"] = "right_turn"
            if (
                _RIGHT_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _RIGHT_TURN_DEGREE_UPPER_BOUND
            ):
                route_index.loc[b, "turn_type"] = "slight_right_turn"
            if (
                _U_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _U_TURN_DEGREE_UPPER_BOUND
            ):
                route_index.loc[b, "turn_type"] = "u_turn"
        left_count = len(route_index[route_index["turn_type"] == "left_turn"])
        slight_left_count = len(route_index[route_index["turn_type"] == "slight_left_turn"])
        right_count = len(route_index[route_index["turn_type"] == "right_turn"])
        slight_right_count = len(route_index[route_index["turn_type"] == "slight_right_turn"])
        u_count = len(route_index[route_index["turn_type"] == "u_turn"])

        travel_time = sum(ox.utils_graph.route_to_gdf(graph, route, "travel_time")["travel_time"])
        total_time = sum(ox.utils_graph.route_to_gdf(graph, route, "total_time")["total_time"])
        total_time_with_turn_penalty = (
            total_time
            + left_turn_penalty * left_count
            + slight_left_turn_penalty * slight_left_count
            + right_turn_penalty * right_count
            + slight_right_turn_penalty * slight_right_count
            + u_turn_penalty * u_count
        )
        distance = sum(ox.utils_graph.route_to_gdf(graph, route, "length")["length"])

    except ValueError:
        # If the path is unsolvable, return -1, -1
        travel_time, total_time, total_time_with_turn_penalty, distance = -1, -1, -1, -1
        (
            route,
            signal_count,
            stop_count,
            crossing_count,
            give_way_count,
            mini_roundabout_count,
            left_count,
            slight_left_count,
            right_count,
            slight_right_count,
            u_count,
        ) = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    return (
        source,
        target,
        sy,
        sx,
        ty,
        tx,
        hod,
        uber_time,
        travel_time,
        total_time,
        total_time_with_turn_penalty,
        distance,
        route,
        signal_count,
        stop_count,
        crossing_count,
        give_way_count,
        mini_roundabout_count,
        left_count,
        slight_left_count,
        right_count,
        slight_right_count,
        u_count,
    )


def run() -> None:
    """Run the routing algorithm for the given OD pairs and save the results to a csv file.

    Returns
    -------
    None

    """
    # number of cpus used for running
    cpus = mp.cpu_count() - 2

    # set up the traffic control penalties (in seconds)
    traffic_time_config = {
        "traffic_signals_time": 0,
        "stop_time": 0,
        "turning_circle_time": 0,
        "crossing_time": 0,
        "give_way_time": 0,
        "mini_roundabout_time": 0,
    }

    od_pair_sample = pd.read_csv(constants.SAMPLED_OD_ROUTES_API_FILE_PATH)

    graph = ox.io.load_graphml(constants.LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH)

    # set up turn penalty dictionary

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    graph = ox.bearing.add_edge_bearings(graph)
    graph = add_edge_control_non_simplified(graph, **traffic_time_config)
    turn_penalty = get_slight_turn_penalty_dict(
        graph,
        left_turn_penalty=0,
        slight_left_turn_penalty=0,
        right_turn_penalty=0,
        slight_right_turn_penalty=0,
        u_turn_penalty=0,
    )

    # choose to calculate the penalized routing
    routing_algorithm = "penalized"
    args = (
        (
            graph,
            od_pair_sample.iloc[i]["oid"],
            od_pair_sample.iloc[i]["did"],
            turn_penalty,
            od_pair_sample.iloc[i]["oy"],
            od_pair_sample.iloc[i]["ox"],
            od_pair_sample.iloc[i]["dy"],
            od_pair_sample.iloc[i]["dx"],
            od_pair_sample.iloc[i]["hod"],
            od_pair_sample.iloc[i]["mean_travel_time"],
            routing_algorithm,
            0,
            0,
            0,
            0,
            0,
        )
        for i in range(len(od_pair_sample))
    )

    cur = time.time()
    pool = mp.Pool(cpus)
    res = pool.starmap_async(calculate, args)
    all_results = res.get()

    pool.close()
    pool.join()
    print(time.time() - cur)

    all_results_df = pd.DataFrame(
        all_results,
        columns=[
            "oid",
            "did",
            "oy",
            "ox",
            "dy",
            "dx",
            "hour_of_day",
            "uber_time",
            "travel_time",
            "total_time",
            "total_time_with_turn_penalty",
            "distance",
            "route",
            "signal_count",
            "stop_count",
            "crossing_count",
            "give_way_count",
            "mini_roundabout_count",
            "left_count",
            "slight_left_count",
            "right_count",
            "slight_right_count",
            "u_count",
        ],
    )

    all_results_df.to_csv(constants.NETWORK_ROUTING_RESULT_FILE_PATH)


if __name__ == "__main__":
    run()
