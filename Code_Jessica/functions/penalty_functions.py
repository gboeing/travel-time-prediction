
def add_edge_traffic_times(G, traffic_signals_time = 30, stop_time = 15, turning_circle_time = 5, crossing_time = 5, give_way_time = 5, mini_roundabout_time = 5):
  
  """
  Calculate traffic time penalties for different features of edge attributes in a graph.

  This function assigns traffic time penalties to edges.
  It considers various highway types, such as traffic signals, stop signs, turning circles, crossings, give-way signs, and mini roundabouts, and assigns corresponding time penalties to these features.

  Parameters
  ----------
  G : networkx.MultiGraph
      An undirected, unprojected graph with 'bearing' attributes on each edge.
  traffic_signals_time : int, optional
      The time penalty for passing through a traffic signal-controlled intersection (default: 30).
  stop_time : int, optional
      The time penalty for stopping at a stop sign or stop-controlled intersection (default: 15).
  turning_circle_time : int, optional
      The time penalty for navigating a turning circle or roundabout (default: 5).
  crossing_time : int, optional
      The time penalty for crossing a pedestrian crossing (default: 5).
  give_way_time : int, optional
      The time penalty for yielding at a give-way or yield sign (default: 5).
  mini_roundabout_time : int, optional
    The time penalty for navigating a mini roundabout (default: 5).

  Returns
  -------
  G : networkx.MultiGraph
      The graph with updated 'traffic_time' and 'total_time' attributes on each edge, representing the calculated traffic-related time penalties.
  """
  
  for u, v, key, data in G.edges(data=True, keys=True):
    traffic_time = 0
    # check if 'highway' tag is in destination node
    if v in G.nodes and 'highway' in G.nodes[v]:
        highway_value = G.nodes[v]['highway']
        # map the time values of traffic panelties
        if highway_value == 'traffic_signals':
            traffic_time = traffic_signals_time
        elif highway_value == 'stop':
            traffic_time = stop_time
        elif highway_value == 'turning_circle':
            traffic_time = turning_circle_time
        elif highway_value == 'crossing':
            traffic_time = crossing_time
        elif highway_value == 'give_way':
            traffic_time = give_way_time
        elif highway_value == 'mini_roundabout':
            traffic_time = mini_roundabout_time

    G[u][v][key]['traffic_time'] = traffic_time
    # calculate 'total_time' by adding 'travel_time' and 'traffic_time'
    # add 'total_time' attribute to the edge
    G[u][v][key]['total_time'] = data.get('travel_time') + traffic_time
  return G




def get_turn_penalty_dict(G, left_turn_penalty = 30, right_turn_penalty = 10, u_turn_penalty = 90):
  """
  Calculate turn penalties for different types of turns at intersections in a graph.

  This function computes turn penalties for various types of turns (left, right, U-turn) at intersections within a road network represented as a graph. 
  The penalties are based on the difference in bearing between the edges that meet at each intersection.

  Parameters
  ----------
  G : networkx.MultiDiGraph
      A directed graph representing a road network with 'bearing' attributes on each edge.
  left_turn_penalty : int, optional
      The penalty for making a left turn at an intersection (default: 30).
  right_turn_penalty : int, optional
      The penalty for making a right turn at an intersection (default: 10).
  u_turn_penalty : int, optional
      The penalty for making a U-turn at an intersection (default: 90).

  Returns
  -------
  penalty : dict
      A dictionary mapping tuples (u, v, m) to their corresponding turn penalties, where:
      - (u, v) represents the incoming road segment,
      - (v, m) represents the outgoing road segment, and
      - The associated value represents the calculated turn penalty based on the difference in bearing.
  """

  penalty = {}
  # iterate all nodes in G
  for v, data in G.nodes(data=True):
    # for each previous node 'u' of node 'v'

    for u, edge_keys in G.pred[v].items():
      # for each edge from 'u' to 'v'
      for key in edge_keys:
        # for each next node 'm' of node 'v'
        for m, next_edge_keys in G[v].items():
          # for each edge from 'v' to 'm'
          for next_key in next_edge_keys:
            # check both edges (u-v, v-m)
            if 'bearing' in G[u][v][key] and 'bearing' in G[v][m][next_key]:
              # calculate the difference between the bearings of the two edges
              bearing_diff = (G[v][m][next_key]['bearing'] - G[u][v][key]['bearing']) % 360
               # add penlties based on the difference
              if  207 < bearing_diff <= 333:
                penalty[(u, v, m)] = left_turn_penalty    # left turn
              elif 27 < bearing_diff <= 153:
                penalty[(u, v, m)] = right_turn_penalty    # right turn
              elif 153 < bearing_diff <= 207:
                penalty[(u, v, m)] = u_turn_penalty    # U turn
              else:
                penalty[(u, v, m)] = 0 # Straight
  return penalty
# time penalty and turn degree referencing r5: https://github.com/conveyal/r5/blob/00e6c8ecffbd0ef5173434b224cd23f3877cdda2/src/main/java/com/conveyal/r5/streets/BasicTraversalTimeCalculator.java#L23



# Source: https://github.com/maxtmng/shortest_path_turn_penalty/
## used the function without any modification
import networkx as nx
from heapq import heappop, heappush
from warnings import warn
from itertools import count
def shortest_path_turn_penalty(G, source, target, weight="weight", penalty={}, next_node = None):
    
  """
    Uses Dijkstra's algorithm to find the shortest weighted paths to one or multiple targets with turn penalty.
    This function is adapted from networkx.algorithms.shortest_paths.weighted._dijkstra_multisource.
    The turn penalty implementation is based on:
    Ziliaskopoulos, A.K., Mahmassani, H.S., 1996. A note on least time path computation considering delays and prohibitions for intersection movements. Transportation Research Part B: Methodological 30, 359–367. https://doi.org/10.1016/0191-2615(96)00001-X
    Parameters
    ----------
    G : NetworkX graph
    source : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.
    target : node label, single node or a list
        Ending node (or a list of ending nodes) for path. Search is halted when any target is found.
    weight: function
        Function with (u, v, data) input that returns that edge's weight
        or None to indicate a hidden edge
    penalty : dict, optional (default={})
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

  G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
  weight = nx.algorithms.shortest_paths.weighted._weight_function(G, weight)
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
      for m,_ in G_succ[source].items():
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
      e = G[v][u]
      for m in G_succ[u]:
          cost = weight(v, u, e)
          if (v,u,m) in penalty:
              cost += penalty[v,u,m]

          if cost is None:
              continue
          vu_dist = dist[v][u] + cost
          if u in dist:
              if m in dist[u]:
                  u_dist = dist[u][m]
                  if vu_dist < u_dist:
                      raise ValueError("Contradictory paths found:", "negative weights?")
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




import osmnx as ox
def get_routes_from_gdfs(G, origins_gdf, destinations_gdf, **kwargs):
  """
  Calculate routes between origins and destinations using a graph with optional turn penalties.
  This function calculates routes between a set of origins and destinations using a given road network graph. 
  It considers optional turn penalties when finding the routes.
  
  Parameters
  ----------
  G : networkx.MultiDiGraph
      A directed graph representing a road network, typically created using OSMnx.
  origins_gdf : geopandas.GeoDataFrame
      A GeoDataFrame containing the origin points with geometry information.
  destinations_gdf : geopandas.GeoDataFrame
      A GeoDataFrame containing the destination points with geometry information.
  **kwargs : keyword arguments
      Additional keyword arguments to be passed to the `shortest_path_turn_penalty` function.
      These can include 'weight' to specify the edge weight attribute and 'penalty' to provide a dictionary of turn penalties.

  Returns
  -------
  routes : list
      A list of routes, where each route is represented as a list of nodes from the origin to the destination.
  """

  routes = []
  for i in range(len(origins_gdf)):
      # find nearest nodes
      orig_node = ox.distance.nearest_nodes(G, origins_gdf.iloc[i]['geometry'].x, origins_gdf.iloc[i]['geometry'].y)
      dest_node = ox.distance.nearest_nodes(G, destinations_gdf.iloc[i]['geometry'].x, destinations_gdf.iloc[i]['geometry'].y)
        
      # find routes while considering penalties
      route = shortest_path_turn_penalty(G, orig_node, dest_node, **kwargs)
      routes.append(route)

  return routes
