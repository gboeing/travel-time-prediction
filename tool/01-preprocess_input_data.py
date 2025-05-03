#!/usr/bin/env python

"""Preprocess the input data."""

import pickle
from pathlib import Path

import constants
import geopandas as gpd
import osmnx as ox
import pandas as pd


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
    """Annotate information to all nodes in the given.

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
    if Path(constants.CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH).is_file():
        with Path.open(constants.CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH, "rb") as file:
            return pickle.load(file)  # noqa: S301

    ox.settings.useful_tags_node = [
        "ref",
        "highway",
        "traffic_signals:direction",
        "direction",
        "crossing",
        "stop",
        "stop:direction",
    ]

    convex_polygon = gpd.GeoDataFrame(geometry=polygon).iloc[0]["geometry"]
    graph_sim = ox.graph_from_polygon(convex_polygon, simplify=True, network_type="drive")
    graph_nonsim = ox.graph_from_polygon(convex_polygon, simplify=False, network_type="drive")

    # Get strongly connected graph
    connected_graph_sim = ox.truncate.largest_component(graph_sim, strongly=True)
    connected_graph_nonsim = ox.truncate.largest_component(graph_nonsim, strongly=True)
    ox.save_graphml(
        connected_graph_nonsim,
        filepath=constants.LA_CLIP_CONVEX_NETWORK_GML_FILE_PATH,
    )

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(connected_graph_sim)
    gdf = gdf_nodes.reset_index(drop=False)
    gdf_proj = ox.projection.project_gdf(gdf, to_latlong=True)
    gdf_proj["x"] = gdf_proj["geometry"].x
    gdf_proj["y"] = gdf_proj["geometry"].y

    # Attach census tract GEOID and uber tract ID to the nodes
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

    gdf_proj_tract_uber.to_csv(constants.CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH)
    with Path.open(constants.CONVEX_STRONGLY_ATTRIBUTES_PICKLE_PATH, "wb") as file:
        pickle.dump(gdf_proj_tract_uber, file)
    return gdf_proj_tract_uber


def sample_od_pairs(
    gdf_proj_tract_uber: gpd.GeoDataFrame,
    sample_size: int | None,
    sample_hour: int,
) -> gpd.GeoDataFrame:
    """Sample OD pairs intersected with uber movement data from the given GeoDataFrame.

    Parameters
    ----------
    gdf_proj_tract_uber: gpd.GeoDataFrame
        Geo Dataframe with census and uber tracts ID attached.
    sample_size: int
        sample size of OD pairs
    sample_hour: int
        sample hour of uber travel time in 24h format

    Returns
    -------
    Sampled ID pairs that in the uber movement with uber travel time attached.

    """
    if Path(constants.SAMPLED_OD_SAMPLE_HOUR_PICKLE_PATH).is_file():
        with Path.open(constants.SAMPLED_OD_SAMPLE_HOUR_PICKLE_PATH, "rb") as file:
            return pickle.load(file)  # noqa: S301

    # Oversample to be filtered later
    origin = (
        gdf_proj_tract_uber.sample(5000000, random_state=123, replace=True)
        .copy()
        .reset_index(drop=True)
    )
    destination = (
        gdf_proj_tract_uber.sample(5000000, random_state=321, replace=True)
        .copy()
        .reset_index(drop=True)
    )

    origin_od = origin[["osmid", "y", "x", "GEOID", "MOVEMENT_ID"]]
    origin_od.columns = ["oid", "oy", "ox", "oGEOID", "oMOVEMENT_ID"]
    destination_od = destination[["osmid", "y", "x", "GEOID", "MOVEMENT_ID"]]
    destination_od.columns = ["did", "dy", "dx", "dGEOID", "dMOVEMENT_ID"]
    sampled_od_pairs = pd.concat([origin_od, destination_od], sort=False, axis=1)

    sampled_od_pairs = sampled_od_pairs.drop(
        sampled_od_pairs[sampled_od_pairs["oid"] == sampled_od_pairs["did"]].index,
    )
    sampled_od_pairs.to_csv(constants.SAMPLED_OD_FILE_PATH)

    # Subset OD pairs that have reference to uber movement 2020
    sampled_od_pairs["oMOVEMENT_ID"] = sampled_od_pairs["oMOVEMENT_ID"].astype(float)
    sampled_od_pairs["dMOVEMENT_ID"] = sampled_od_pairs["dMOVEMENT_ID"].astype(float)
    sampled_od_pairs["uber_OD"] = list(
        zip(sampled_od_pairs.oMOVEMENT_ID, sampled_od_pairs.dMOVEMENT_ID, strict=False),
    )

    uber_2020_travel_time = pd.read_csv(
        constants.UBER_TRAVEL_TIME_FILE_PATH,
        dtype={"sourceid": float, "dstid": float},
    )
    # Select the OD pairs that have reference in uber movement 2020
    unique_od = set(zip(uber_2020_travel_time.sourceid, uber_2020_travel_time.dstid, strict=False))
    od_pairs_uber = sampled_od_pairs[sampled_od_pairs["uber_OD"].isin(unique_od)]
    # Save all the sampled 1,197,651 OD pairs that have a reference in uber movement 2020 to csv
    od_pairs_uber.to_csv(constants.SAMPLED_OD_ALL_UBER_FILE_PATH)
    # deduplicate
    od_pairs_uber_dedup = od_pairs_uber.drop_duplicates(subset=["oid", "did"], keep="first")

    uber_2020_travel_time["uber_OD"] = list(
        zip(uber_2020_travel_time.sourceid, uber_2020_travel_time.dstid, strict=False),
    )
    uber_2020_travel_time["uber_OD"] = uber_2020_travel_time["uber_OD"].astype(str)
    od_pairs_uber_dedup["uber_OD"] = od_pairs_uber_dedup["uber_OD"].astype(str)

    # Merge sampled OD pairs with uber movement travel time result
    uber_dedup_merge = od_pairs_uber_dedup.merge(uber_2020_travel_time, how="left", on="uber_OD")

    # If an OD pairs have multiple hour of day travel time, sample one hour of day
    uber_dedup_merge_sample = (
        uber_dedup_merge.groupby(["oid", "did"])
        .apply(lambda x: x.sample(1, random_state=123))
        .reset_index(drop=True)
    )

    sampled_hour_df = (
        uber_dedup_merge_sample[uber_dedup_merge_sample["hod"] == sample_hour]
        .sample(
            n=len(uber_dedup_merge_sample) if sample_size is None else sample_size,
            random_state=123,
        )
        .copy()
    )
    sampled_hour_df.to_csv(constants.SAMPLED_OD_SAMPLE_HOUR_FILE_PATH)
    with Path.open(constants.SAMPLED_OD_SAMPLE_HOUR_PICKLE_PATH, "wb") as file:
        pickle.dump(sampled_hour_df, file)
    return sampled_hour_df


if __name__ == "__main__":
    la_ucdb, la_tract, ca_tract, uber_tract = get_gdf_inputs()
    la_convex = create_convex_hull(la_ucdb, la_tract)
    gdf_proj_tract = get_annotated_graph(la_convex, ca_tract, uber_tract)
    sample_od_pairs(gdf_proj_tract, None, 3)
