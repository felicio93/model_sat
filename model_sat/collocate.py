import logging
from typing import Union, Optional, List

import numpy as np
import scipy
import xarray as xr
import ocsmesh

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print logs to the console
        # logging.FileHandler('get_sat.log', mode='w')
    ]
)
_logger = logging.getLogger()


def convert_longitude(lon: Union[float, np.ndarray],
                      mode: int = 1) -> np.ndarray:
    """
    Convert longitudes between common geographic conventions.

    Args:
        lon: array-like of longitudes
        mode: conversion mode:
            - 1: Convert from [-180, 180] to [0, 360] (Greenwich at 0°)
            - 2: Convert from [0, 360] to [-180, 180] (Greenwich at 0°)
            - 3: Convert from [-180, 180] to [0, 360] (Greenwich at 180°)

    Returns:
        np.ndarray of converted longitudes
    """
    _logger.debug("Converting longitude with mode %d", mode)
    lon = np.asarray(lon)
    if mode == 1:
        return lon % 360
    elif mode == 2:
        return np.where(lon > 180, lon - 360, lon)
    elif mode == 3:
        return lon + 180
    return lon


def inverse_distance_weights(dists: np.ndarray,
                             power: float = 1.) -> np.ndarray:
    """
    Compute normalized inverse distance weights.

    Args:
        dists: 2D array of distances [n_points, n_neighbors]
        power: exponent applied to inverse weighting (default=1.0)

    Returns:
        2D array of normalized weights with same shape as dists
    """
    _logger.debug("Calculating inverse distance weights with power %f", power)

    weights = 1 / np.power(np.maximum(dists, 1e-6), power)
    return weights / weights.sum(axis=1, keepdims=True)


def temporal_collocation_nearest(ds_sat: xr.Dataset,
                                 model_times: np.ndarray[np.datetime64],
                                 time_buffer: np.timedelta64,
                                 ) -> tuple:
    """
    Find the nearest model output times to satellite observation times.

    Args:
        ds_sat: xarray Dataset with satellite data (must contain 'time' variable)
        model_times: array of model output times (e.g., model_ds['time'].values)
        time_buffer: time margin around model times
                     to include satellite observations
                     (e.g., np.timedelta64(30, 'm') for ±30 minutes)

    Returns:
        ds_sat_subset: satellite data filtered within the model time range
        nearest_model_times: array of model times nearest to each
                             satellite time
        time_dt: array of time differences (in seconds) between
                 satellite and model times
    """
    _logger.info("Performing temporal collocation (nearest)")
    start_date = model_times.min() - time_buffer
    end_date = model_times.max() + time_buffer
    ds_sat_subset = ds_sat.sortby('time').sel(time=slice(start_date, end_date))

    if len(ds_sat_subset['time']) == 0:
        _logger.error("No satellite data found in time range %s to %s",
                      start_date, end_date)
        raise ValueError(f"No satellite data found in range \
                         {start_date} to {end_date}")

    sat_times = ds_sat_subset['time'].values
    nearest_time_indices = np.abs(sat_times[:, None] - model_times[None, :]).argmin(axis=1)
    nearest_model_times = model_times[nearest_time_indices]
    time_dt = (sat_times - nearest_model_times).astype('timedelta64[s]').astype(int)

    _logger.info("Temporal collocation complete")
    return ds_sat_subset, nearest_model_times, time_dt


def temporal_collocation_interpolated(ds_sat: xr.Dataset,
                                      model_times: np.ndarray[np.datetime64],
                                      time_buffer: np.timedelta64,
                                      ) -> tuple:
    """
    Perform temporal collocation using linear interpolation
    between model time steps.

    Args:
        ds_sat: xarray Dataset with satellite data (must contain 'time' variable)
        model_times: array of model output times (e.g., model_ds['time'].values)
        time_buffer: time margin around model times to include
                     satellite observations (e.g., np.timedelta64(30, 'm'))

    Returns:
        ds_sat_subset: satellite data filtered within the model time range
        indices_before: indices of the model time steps just
                        before each satellite time
        indices_after: indices of the model time steps just
                       after each satellite time
        weights: interpolation weights between before/after model steps
        nearest_model_times: model time (either before or after) closest
                             to each satellite time
        time_dt: time difference (in seconds) between satellite and
                 closest model time
    """
    _logger.info("Performing temporal collocation (interpolated)")
    start_date = model_times.min() - time_buffer
    end_date = model_times.max() + time_buffer
    ds_sat_subset = ds_sat.sortby('time').sel(time=slice(start_date, end_date))

    if len(ds_sat_subset['time']) == 0:
        _logger.error("No satellite data found in time range %s to %s",
                      start_date, end_date)
        raise ValueError("No satellite data in time buffer range.")

    sat_times = ds_sat_subset['time'].values
    model_times_np = model_times.astype('datetime64[s]')
    indices_before = np.searchsorted(model_times_np,
                                     sat_times, side='right') - 1
    indices_after = indices_before + 1

    indices_before = np.clip(indices_before, 0, len(model_times) - 2)
    indices_after = np.clip(indices_after, 1, len(model_times) - 1)

    t0 = model_times_np[indices_before]
    t1 = model_times_np[indices_after]
    weights = (sat_times - t0) / (t1 - t0)

    dt0 = np.abs(sat_times - t0)
    dt1 = np.abs(sat_times - t1)
    nearest_model_times = np.where(dt0 <= dt1, t0, t1)

    time_dt = (sat_times - nearest_model_times).astype('timedelta64[s]').astype(int)

    _logger.info("Temporal collocation complete")
    return ds_sat_subset, indices_before, indices_after, weights, nearest_model_times, time_dt


def spatial_collocation(ds_sat_subset: xr.Dataset,
                        tree: scipy.spatial.cKDTree,
                        n_nearest: int) -> tuple:
    """
    Find nearest model nodes via KDTree search.

    Args:
        ds_sat_subset: xarray Dataset containing satellite data with 
                       'lon' and 'lat' variables
        tree: cKDTree built from model grid node coordinates
        n_nearest: number of nearest model nodes to select

    Returns:
        dists: 2D array of distances from satellite points to nearest 
               model nodes [n_points, n_nearest]
        inds: 2D array of corresponding node indices [n_points, n_nearest]
    """
    _logger.info("Performing spatial collocation")
    dists, inds = tree.query(
        np.column_stack((ds_sat_subset['lon'].values,
                         ds_sat_subset['lat'].values)), 
        k=n_nearest
    )
    _logger.info("Spatial collocation complete")
    return dists, inds


def extract_model_data(m_file: xr.DataArray,
                       model_depth: np.ndarray,
                       times_or_inds: np.ndarray,
                       nearest_nodes: np.ndarray,
                       interpolate: bool = False,
                       inds_after: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None) -> tuple:
    """
    Extract model values and depths at matched or
    interpolated times and spatial nodes.

    Args:
        m_file: DataArray of model variable (e.g., significant wave height)
        model_depth: array of model node depths
        times_or_inds: array of either datetime64 values (if interpolate=False)
                       or indices of the first time step (if interpolate=True)
        nearest_nodes: 2D array of node indices [n_points, n_nearest]
        interpolate: whether to perform linear time
                     interpolation (default=False)
        inds_after: indices of the time step after each
                    satellite pass (required if interpolate=True)
        weights: interpolation weights between
                 times (required if interpolate=True)

    Returns:
        values: 2D array of extracted model variable [n_points, n_nearest]
        depths: 2D array of model depths at the same nodes [n_points, n_nearest]

    Notes:
        This works for SCHISM only, may add other models in the future
    """
    _logger.info("Performing model data extraction")
    values, depths = [], []

    for i, nodes in enumerate(nearest_nodes):
        if interpolate:
            ib, ia = times_or_inds[i], inds_after[i]
            w = weights[i]
            v0 = m_file.isel(time=ib, nSCHISM_hgrid_node=nodes).values
            v1 = m_file.isel(time=ia, nSCHISM_hgrid_node=nodes).values
            values.append(v0 * (1 - w) + v1 * w)
        else:
            t = times_or_inds[i]
            values.append(m_file.sel(time=t, nSCHISM_hgrid_node=nodes).values)

        depths.append(model_depth[nodes])

    _logger.info("Model data extraction complete")
    return np.array(values), np.array(depths)


def collocate_data(model_files: list[xr.DataArray],
                   ds_sat: xr.Dataset,
                   mesh_x: np.ndarray,
                   mesh_y: np.ndarray,
                   mesh_depth: np.ndarray,
                   dist_coast: xr.Dataset,
                   n_nearest: int = 3,
                   time_buffer: np.timedelta64 = np.timedelta64(30, 'm'),
                   weight_power: float = 1.0,
                   temporal_interp: bool = False,
                   output_path: str | None = None
                   ) -> xr.Dataset:

    """
    Main collocation routine to align satellite and model data in space and time.

    Args:
        model_files: list of xarray DataArrays of model outputs (e.g., SWH)
        ds_sat: xarray Dataset with satellite observations
        mesh_x: array of model grid node longitudes
        mesh_y: array of model grid node latitudes
        mesh_depth: array of model node depths
        dist_coast: xarray Dataset containing 'distcoast' variable with 
                    distance from the coast
        n_nearest: number of nearest model nodes to consider (default=3)
        time_buffer: time margin around model data range to search 
                     satellite data (default=30 minutes)
        weight_power: exponent for inverse distance weighting (default=1.0)
        temporal_interp: if True, use linear time interpolation; 
                         if False, use nearest timestep (default=False)

    Returns:
        xarray.Dataset with collocated satellite and model data, including:
            - satellite and model SWH
            - model depths at collocation points
            - spatial distances to nearest nodes
            - time deltas between sat and model
            - weighted spatial average
            - biases (raw and weighted)
            - distance from coast
            - metadata (lat/lon/time/source)
    """

    _logger.info("Starting collocation process")
    tree = scipy.spatial.cKDTree(np.column_stack((mesh_x, mesh_y)))
    distcoast = dist_coast['distcoast']  # Keep as xarray DataArray

    results = {
        'sat_swh': [], 'model_swh': [], 'model_dpt': [],
        'dist_deltas': [], 'node_ids': [], 'time_deltas': [],
        'bias_raw': [], 'bias_weighted': [], 'dist_coast': [], 
        'source_sat': [], 'time_sat': [], 'lat_sat': [], 'lon_sat': [],
        'model_swh_weighted': []
    }

    for m_file in model_files:
        _logger.info("Processing model file: %s", m_file)
        if temporal_interp:
            # Perform temporal interpolation
            ds_sat_subset, ib, ia, weights, nearest_model_times, time_dt = temporal_collocation_interpolated(
                ds_sat, m_file['time'].values, time_buffer
            )
            dists, inds = spatial_collocation(ds_sat_subset, tree, n_nearest)
            nearest_model_values, nearest_model_depths = extract_model_data(
                m_file, mesh_depth, ib, inds, interpolate=True, inds_after=ia,
                weights=weights
            )
        else:
            # Perform nearest time matching
            ds_sat_subset, nearest_model_times, time_dt = temporal_collocation_nearest(
                ds_sat, m_file['time'].values, time_buffer
            )
            dists, inds = spatial_collocation(ds_sat_subset, tree, n_nearest)
            nearest_model_values, nearest_model_depths = extract_model_data(
                m_file, mesh_depth, nearest_model_times, inds, interpolate=False
            )

        # Weighted spatial average using inverse distance
        weights = inverse_distance_weights(dists, power=weight_power)
        model_swh_weighted = (nearest_model_values * weights).sum(axis=1)

        # Coast distance
        lats = ds_sat_subset['lat'].values
        lons = convert_longitude(ds_sat_subset['lon'].values, 1)
        dist_coast_select = distcoast.sel(
            latitude=xr.DataArray(lats, dims='points'),
            longitude=xr.DataArray(lons, dims='points'),
            method='nearest'
        ).values

        results['sat_swh'].append(ds_sat_subset.swh.values)
        results['model_swh'].append(nearest_model_values)
        results['model_dpt'].append(nearest_model_depths)
        results['dist_deltas'].append(dists)
        results['node_ids'].append(inds)
        results['time_deltas'].append(time_dt)
        results['model_swh_weighted'].append(model_swh_weighted)
        results['bias_raw'].append(nearest_model_values.mean(axis=1) - ds_sat_subset.swh.values)
        results['bias_weighted'].append(model_swh_weighted - ds_sat_subset.swh.values)
        results['dist_coast'].append(dist_coast_select)
        results['time_sat'].append(ds_sat_subset['time'].values)
        results['lat_sat'].append(lats)
        results['lon_sat'].append(ds_sat_subset['lon'].values)
        results['source_sat'].append(ds_sat_subset['source'].values)

    _logger.info("Collocation complete, saving output")
    ds_out = xr.Dataset(
        {
            'sat_swh': (['time_sat'],
                        np.concatenate(results['sat_swh'])),
            'model_swh': (['time_sat',
                           'nearest_nodes'],
                          np.vstack(results['model_swh'])),
            'model_swh_weighted': (['time_sat'],
                                   np.concatenate(results['model_swh_weighted'])),
            'model_dpt': (['time_sat', 'nearest_nodes'],
                          np.vstack(results['model_dpt'])),
            'dist_deltas': (['time_sat', 'nearest_nodes'],
                            np.vstack(results['dist_deltas'])),
            'node_ids': (['time_sat', 'nearest_nodes'],
                         np.vstack(results['node_ids'])),
            'time_deltas': (['time_sat'],
                            np.concatenate(results['time_deltas'])),
            'bias_raw': (['time_sat'],
                         np.concatenate(results['bias_raw'])),
            'bias_weighted': (['time_sat'],
                              np.concatenate(results['bias_weighted'])),
            'dist_coast': (['time_sat'],
                           np.concatenate(results['dist_coast'])),
            'source_sat': (['time_sat'],
                           np.concatenate(results['source_sat'])),
        },
        coords={
            'time_sat': np.concatenate(results['time_sat']),
            'lat_sat': np.concatenate(results['lat_sat']),
            'lon_sat': np.concatenate(results['lon_sat']),
            'nearest_nodes': np.arange(n_nearest),
        }
    )

    if output_path:
        if output_path.endswith('.nc'):
            ds_out.to_netcdf(output_path)
            _logger.info(f"Saved collocated dataset to NetCDF: {output_path}")
        elif output_path.endswith('.parquet'):
            df = ds_out.to_dataframe().reset_index()
            df.to_parquet(output_path, index=False)
            _logger.info(f"Saved collocated dataset to Parquet: {output_path}")
        else:
            _logger.error("Unsupported output format. Use .nc or .parquet")

    return ds_out
