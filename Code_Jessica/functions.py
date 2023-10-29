
def add_edge_traffic_times(G, traffic_signals_time = 2, stop_time = 2, turning_circle_time = 0, crossing_time = 1.5, give_way_time = 1.5, mini_roundabout_time = 1.5):

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
