"""Preprocess the input data."""

import geopandas as gpd
import osmnx as ox

import constants


def get_gdf_inputs() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Get the input data in GeoDataFrames with path specified in constants.py.

    Returns
    -------
        Tuple of GeoDataFrames with same CRS.

    """
    ucdb_gdf = gpd.read_file(constants.UCDB_FILE_PATH)
    ca_tract_gdf = gpd.read_file(constants.CA_TRACT_FILE_PATH)
    uber_tract_gdf = gpd.read_file(constants.UBER_TRACT_FILE_PATH)

    # Project the CRS to be the same
    ca_tract_gdf = ca_tract_gdf.to_crs(ucdb_gdf.crs)
    uber_tract_gdf = uber_tract_gdf.to_crs(ucdb_gdf.crs)

    # Select LA only
    la_ucdb_gdf = ucdb_gdf[ucdb_gdf["UC_NM_MN"] == "Los Angeles"]
    la_tract_gdf = ca_tract_gdf[ca_tract_gdf["COUNTYFP"] == "037"]

    return la_ucdb_gdf, la_tract_gdf, ca_tract_gdf, uber_tract_gdf


def create_convex_hull(
    ucdb_gdf: gpd.GeoDataFrame,
    tract_gdf: gpd.GeoDataFrame,
) -> gpd.geoseries.GeoSeries:
    """Create convex hull polygon from input gdf.

    Parameters
    ----------
    ucdb_gdf: gpd.GeoDataFrame
        GHS UCDB in GeoDataFrame
    tract_gdf: gpd.GeoDataFrame
        census tracts in GeoDataFrame

    Returns
    -------
    A convex hull of these two's spatial intersection.

    """
    # Get the intersection proportion of these two shapefiles
    clip_gdf = gpd.clip(tract_gdf, ucdb_gdf).dissolve()

    # Get the convex hull and save to file
    clip_convex = clip_gdf.convex_hull
    clip_convex.to_file(constants.LA_CLIP_CONVEX_FILE_PATH)

    return clip_convex


def get_annotated_graph(
    polygon: gpd.geoseries.GeoSeries,
    ca_tract_gdf: gpd.GeoDataFrame,
    uber_tract_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Annotate information to all nodes in the given .

    Parameters
    ----------
    polygon: gpd.geoseries.GeoSeries
        input polygon to query street network
    ca_tract_gdf: gpd.GeoDataFrame
        census tracts in GeoDataFrame
    uber_tract_gdf: gpd.GeoDataFrame
        uber tracts in GeoDataFrame

    Returns
    -------
    Nodes of query street network with census and uber tracts ID attached.

    """
    convex_polygon = gpd.GeoDataFrame(geometry=polygon).iloc[0]["geometry"]
    graph = ox.graph_from_polygon(convex_polygon, network_type="drive")

    # Get strongly connected graph
    connected_graph = ox.truncate.largest_component(graph, strongly=True)
    ox.save_graphml(connected_graph, filepath=constants.LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(connected_graph)
    gdf = gdf_nodes.reset_index(drop=False)
    gdf_proj = ox.projection.project_gdf(gdf, to_latlong=True)
    gdf_proj["x"] = gdf_proj["geometry"].x
    gdf_proj["y"] = gdf_proj["geometry"].y

    # Attach information on
    selected_cols = ["osmid", "x", "y", "highway", "street_count", "ref", "geometry", "GEOID"]
    uber_only_cols = ["MOVEMENT_ID", "TRACT"]
    gdf_proj_tract_origin = gpd.sjoin(gdf_proj, ca_tract_gdf, how="left", predicate="within")[
        selected_cols
    ]
    gdf_proj_tract_uber = gpd.sjoin(
        gdf_proj_tract_origin,
        uber_tract_gdf,
        how="left",
        predicate="within",
    )[selected_cols + uber_only_cols].drop_duplicates(subset=["osmid"], keep="first")

    gdf_proj_tract_uber.to_csv(constants.CONVEX_STRONGLY_ATTRIBUTES_FILE_PATH)
    return gdf_proj_tract_uber


if __name__ == "__main__":
    la_ucdb, la_tract, ca_tract, uber_tract = get_gdf_inputs()
    la_convex = create_convex_hull(la_ucdb, la_tract)
    gdf_proj_tract = get_annotated_graph(la_convex, ca_tract, uber_tract)
    print(gdf_proj_tract.head())
