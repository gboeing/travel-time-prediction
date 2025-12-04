#!/usr/bin/env python

"""Calculating the travel time and distance of OD in a graph based on routing algorithms."""

import logging
import multiprocessing as mp
import warnings
from collections import Counter
from heapq import heappop, heappush
from itertools import count
from os import getenv

import constants
import networkx as nx
import osmnx as ox
import pandas as pd
from networkx.exception import NetworkXNoPath, NodeNotFound

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


def control_applies(graph: nx.MultiGraph, u: int, v: int, key: int, tag_name: str) -> bool:
    """Judge whether the traffic control applies based on the direction.

    Return True if the control tag `tag_name` at node v should apply
    on the edge (u, v, key), considering any tag-specific or general direction.

    Parameters
    ----------
    graph : nx.MultiDiGraph
      The OSMnx graph containing node and edge attributes.
    u : int
      Source node ID of the edge.
    v : int
      Targfet node ID of the edge (where the traffic controls are).
    key : int
      Edge key for parallel edges
    tag_name : str
      Name of the control tag with directions, e.g. "traffic_signals", "stop", "give_way".

    Returns
    -------
    bool
      True if the control tag `tag_name` should apply on the edge (u, v, key).
      False if the control tag `tag_name` 's direction should not apply on the edge (u, v, key).

    """
    field = f"{tag_name}:direction"
    dir_val = graph.nodes[v][field] if field in graph.nodes[v] else graph.nodes[v].get("direction")
    if dir_val not in ("forward", "backward"):
        return True
    rev = graph[u][v][key]["reversed"]
    if dir_val == "forward" and rev:
        return False
    return not (dir_val == "backward" and not rev)


def add_edge_control_non_simplified(  # noqa: PLR0913
    graph: nx.MultiGraph,
    traffic_signals_time: float = 2,
    stop_time: float = 2,
    crossing_time: float = 1.5,
    give_way_time: float = 1.5,
    mini_roundabout_time: float = 1.5,
) -> nx.MultiGraph:
    """Calculate traffic time penalties for different features of edge attributes in a graph.

    It assigns traffic controls to edges, considering nodes with multiple highway tags.
    It considers traffic signals, stop signs, crossings, give-way signs, and mini roundabouts,
    and assigns corresponding time penalties specified to these features.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
      An undirected, unprojected graph with 'bearing' attributes on each edge.
    traffic_signals_time : float, optional
      The time penalty for passing through a traffic signal-controlled intersection (default: 2).
    stop_time : float, optional
      The time penalty for stopping at a stop sign or stop-controlled intersection (default: 2).
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
    for u, v, key, _data in graph.edges(data=True, keys=True):
        traffic_time = 0
        controls = []

        if "highway" in graph.nodes[v]:
            # check if 'highway' tag is in destination node
            raw_value = graph.nodes[v]["highway"]
            if isinstance(raw_value, str):
                tags = [t.strip() for t in raw_value.split(";")]
                # if there are multiple tags in the node, split them into a list
                # or if there is only one tag in the node, also put it as a list
            elif isinstance(raw_value, list):
                # if it is already a list, then pass it along
                tags = raw_value
            else:
                tags = []
                # if not, it means that it has no tag, then pass it with an empty list
            # add traffic controls and assigned penalties on edges
            if "traffic_signals" in tags and control_applies(graph, u, v, key, "traffic_signals"):
                traffic_time += traffic_signals_time
                controls.append("traffic_signals")
            if "stop" in tags and control_applies(graph, u, v, key, "stop"):
                traffic_time += stop_time
                controls.append("stop")
            if "crossing" in tags:
                traffic_time += crossing_time
                controls.append("crossing")
            if "give_way" in tags and control_applies(graph, u, v, key, "give_way"):
                traffic_time += give_way_time
                controls.append("give_way")
            if "mini_roundabout" in tags:
                traffic_time += mini_roundabout_time
                controls.append("mini_roundabout")

        graph[u][v][key]["traffic_time"] = traffic_time
        graph[u][v][key]["traffic_control"] = ";".join(controls)
        # calculate 'total_time' by adding 'travel_time' and 'traffic_time'
        # add 'total_time' attribute to the edge
        graph[u][v][key]["total_time"] = graph[u][v][key]["travel_time"] + traffic_time
    return graph


def get_slight_turn_penalty_dict(  # noqa: C901, PLR0913
    graph: nx.MultiDiGraph,
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
    graph : networkx.MultiDiGraph
      A directed graph representing a road network with 'bearing' attributes on each edge.
    left_turn_penalty : float, optional
      The penalty (seconds) for making a left turn at an intersection (default: 30).
    slight_left_turn_penalty : float, optional
      The penalty (seconds) for making a slight left turn at an intersection (default: 20).
    right_turn_penalty : float, optional
      The penalty (seconds) for making a right turn at an intersection (default: 10).
    slight_right_turn_penalty : float, optional
      The penalty (seconds) for making a slight right turn at an intersection (default: 20).
    u_turn_penalty : float, optional
      The penalty (seconds) for making a U-turn at an intersection (default: 90).

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
def shortest_path_turn_penalty(  # noqa: C901, PLR0912, PLR0913, PLR0915
    graph: nx.MultiDiGraph,
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
    graph : networkx.MultiDiGraph
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
    weight = nx.algorithms.shortest_paths.weighted._weight_function(graph, weight)  # noqa: SLF001
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
    if reached_target is None:
        # no directed path reached the target
        raise nx.NetworkXNoPath(source, target)

    return paths[reached_target]
    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.


def calculate(  # noqa: C901, PLR0913, PLR0912
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
    left_turn_penalty: float = 0,
    slight_left_turn_penalty: float = 0,
    right_turn_penalty: float = 0,
    slight_right_turn_penalty: float = 0,
    u_turn_penalty: float = 0,
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
    left_turn_penalty : float (default: 0s)
      Penalty in seconds for left turns (225-315 degrees), set to 0
    slight_left_turn_penalty : float (default: 0s)
      Penalty in seconds for slight left turns (315 - 330 degrees), set to 0
    right_turn_penalty : float (default: 0s)
      Penalty in seconds for right turns (45-135 degrees), set to 0
    slight_right_turn_penalty : float (default: 0s)
      Penalty in seconds for slight right turns (30-45 degrees), set to 0
    u_turn_penalty : float (default: 0s)
      Penalty in seconds for u turns (135-225 degrees), set for 0s

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
    # immediate return if either source or target code is missing
    if not graph.has_node(source) or not graph.has_node(target):
        return (
            source,
            target,
            sy,
            sx,
            ty,
            tx,
            hod,
            uber_time,
            -1,
            -1,
            -1,
            -1,
            [],
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        )
    if algorithm not in {"penalized", "freeflow", "shortest_distance"}:
        msg = f"Unknown algorithm: {algorithm}"
        raise ValueError(msg)

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
        else:
            # otherwise, just use shortest path
            route = shortest_path_turn_penalty(graph, source, target, weight="length", penalty=None)

        route_edges = ox.routing.route_to_gdf(graph, route)
        signal_count = route_edges["traffic_control"].str.contains("traffic_signals").sum()
        stop_count = route_edges["traffic_control"].str.contains("stop").sum()
        crossing_count = route_edges["traffic_control"].str.contains("crossing").sum()
        give_way_count = route_edges["traffic_control"].str.contains("give_way").sum()
        mini_roundabout_count = route_edges["traffic_control"].str.contains("mini_roundabout").sum()

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
                _SLIGHT_RIGHT_TURN_DEGREE_LOWER_BOUND
                < route_index.loc[b, "turn_degree"]
                <= _SLIGHT_RIGHT_TURN_DEGREE_UPPER_BOUND
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

        travel_time = sum(
            ox.routing.route_to_gdf(graph, route, weight="travel_time")["travel_time"],
        )
        total_time = sum(ox.routing.route_to_gdf(graph, route, weight="total_time")["total_time"])
        total_time_with_turn_penalty = (
            total_time
            + left_turn_penalty * left_count
            + slight_left_turn_penalty * slight_left_count
            + right_turn_penalty * right_count
            + slight_right_turn_penalty * slight_right_count
            + u_turn_penalty * u_count
        )
        distance = sum(ox.routing.route_to_gdf(graph, route, weight="length")["length"])

    except (ValueError, NodeNotFound, NetworkXNoPath, KeyError):
        # If the path is unsolvable, return -1, -1
        route = []
        travel_time, total_time, total_time_with_turn_penalty, distance = -1, -1, -1, -1
        (
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
        ) = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
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


def run_single_network() -> None:
    """Run the routing workflow for a single network.

    Parameters
    ----------
    network_path : str
        Path to the GraphML file of the clipped OSM network.
    traffic_control_path : str
        Output CSV path for the traffic control summary table.
    routing_result_path : str
        Output CSV path for the OD routing results used for modeling.

    """
    # number of cpus used for running

    logger = logging.getLogger("tool")
    cpus = mp.cpu_count() - 6

    network_path = str(constants.ROUTING_NETWORK_READ_PATH)
    routing_result_path = str(constants.ROUTING_RESULT_FILE_PATH)
    traffic_control_path = str(constants.TRAFFIC_CONTROL_FILE_PATH)

    logger.info("Loading Network: %s", network_path)
    logger.info("Saving Results to: %s", routing_result_path)

    # set up the traffic control penalties (in seconds)
    traffic_time_config = {
        "traffic_signals_time": 0,
        "stop_time": 0,
        "crossing_time": 0,
        "give_way_time": 0,
        "mini_roundabout_time": 0,
    }

    od_pair_sample = pd.read_csv(constants.SAMPLED_OD_ROUTES_API_FILE_PATH)

    if getenv("TTP_TEST") == "true":
        logger.warning("TEST MODE ACTIVATED: Only processing first 500 rows!")
        od_pair_sample = od_pair_sample.head(500)

    graph = ox.io.load_graphml(network_path)
    resim_graph = ox.simplification.simplify_graph(graph)
    simplify_nodes, simplify_edges = ox.graph_to_gdfs(resim_graph)
    # statistics for table 1
    tag_list = ["traffic_signals", "stop", "crossing", "give_way", "mini_roundabout"]
    tag_counter = Counter()

    for _, data in graph.nodes(data=True):
        highway = data.get("highway")
        tags = []
        if isinstance(highway, str):
            tags = [t.strip() for t in highway.split(";")]
            # if there are multiple tags in the node, split them into a list
            # or if there is only one tag in the node, also put it as a list
        elif isinstance(highway, list):
            # if it is already a list, then pass it along
            tags = highway
        for tag in tags:
            if tag in tag_list:
                tag_counter[tag] += 1
    total_controls = sum(tag_counter.values())
    total_nodes = len(graph.nodes)
    total_intersections = len(simplify_nodes[simplify_nodes["street_count"] > 1])

    # Create table data
    tc_rows = [
        {"Element": "Crossing", "Count": tag_counter.get("crossing", 0)},
        {"Element": "Stop sign", "Count": tag_counter.get("stop", 0)},
        {"Element": "Traffic signal", "Count": tag_counter.get("traffic_signals", 0)},
        {"Element": "Mini roundabout", "Count": tag_counter.get("mini_roundabout", 0)},
        {"Element": "Give way", "Count": tag_counter.get("give_way", 0)},
        {"Element": "Total traffic control elements", "Count": total_controls},
        {"Element": "Total street intersections", "Count": total_intersections},
        {"Element": "Total nodes", "Count": total_nodes},
    ]
    tc_df = pd.DataFrame(tc_rows)
    tc_df.to_csv(traffic_control_path, index=False)

    valid_nodes = set(graph.nodes)

    # Keep only rows where both oid and did are in valid_nodes
    od_pair_sample = od_pair_sample[
        od_pair_sample["oid"].isin(valid_nodes) & od_pair_sample["did"].isin(valid_nodes)
    ]
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
    routing_algorithm = "freeflow"
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

    ctx = mp.get_context("spawn")
    with ctx.Pool(cpus) as pool:
        res = pool.starmap_async(calculate, args)
        all_results = res.get()

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

    all_results_df.to_csv(routing_result_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_single_network()
