import os
import requests
import logging
import shutil

import xarray as xr

from datetime import datetime, timedelta
from typing import (Union, Optional,List)

from urls import URL_TEMPLATES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


def generate_daily_dates(start_date_str: str, end_date_str: str) -> List[str]:
    """
    This function generates a list of formated
    dates between the start and end dates.

    Args:
        start_date_str: String with dates as 'YYYY-MM-DD'
        end_date_str: String with dates as 'YYYY-MM-DD'

    Returns:
        List of formated dates (daily).
    """

    start_date = datetime.strptime(start_date_str,
                                   '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str,
                                 '%Y-%m-%d')

    return [(start_date + timedelta(days=i)).strftime('%Y%m%d')
            for i in range((end_date - start_date).days + 1)]


def download_sat_file(url: str, save_path: str) -> None:
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        _logger.info(f"Downloaded {os.path.basename(save_path)}")
    except requests.RequestException as e:
        _logger.warning(f"Failed to download {url}: {type(e).__name__} - {e}")


def crop_data(dataset: xr.Dataset,
              lat_min: float,
              lat_max: float,
              lon_min: float,
              lon_max: float) -> xr.Dataset:
    """
    Crops xarray data based on lats and lons

    Args:
        lat_min: float/int of mininum latitude
        lat_max: float/int of maximum latitude
        lon_min: float/int of mininum longitude
        lon_max: float/int of maximum latitude
    Returns:
        xarray object of the cropped data
    Note:
        Satellite data uses the -180 to 180 standard
        If you want to cross the meridian, then pass a lon_min > lon_max
        
    """
    # Check if latitude and longitude coordinates are in the dataset
    if 'lat' not in dataset or 'lon' not in dataset:
        raise ValueError("Dataset does not contain lat or lon dimensions")

    if lon_min < lon_max:
        lon_mask = (dataset.lon >= lon_min) & (dataset.lon <= lon_max)
    else:
        lon_mask = (dataset.lon >= lon_min) | (dataset.lon <= lon_max)

    lat_mask = (dataset.lat >= lat_min) & (dataset.lat <= lat_max)
    cropped = dataset.where(lat_mask & lon_mask, drop=True)

    return cropped


def get_sat(start_date: str,
         end_date: str,
         sat: str,
         output_dir: Union[str, os.PathLike],
         lat_min: Optional[float] = None,
         lat_max: Optional[float] = None,
         lon_min: Optional[float] = None,
         lon_max: Optional[float] = None,
         concat: bool = True,
         clean_raw: bool = False,
         clean_cropped: bool = False) -> Optional[xr.Dataset]:
    """
    Download, crop, and optionally concatenate satellite data.

    Args:
        start_date: Start date in 'YYYY-MM-DD'
        end_date: End date in 'YYYY-MM-DD'
        sat: Satellite key for URL_TEMPLATES
        output_dir: Directory to save files
        lat_min, lat_max, lon_min, lon_max: Optional cropping bounds
        concat: Save a single concatenated output
        clean_raw: Delete raw files after processing
        clean_cropped: Delete cropped files after processing

    Returns:
        xarray.Dataset if concatenated, otherwise None
    """
    try:
        url_template = URL_TEMPLATES[sat]
    except KeyError:
        raise ValueError(f"Unknown satellite key: {sat}")

    output_dir = os.path.join(output_dir, sat)
    raw_dir = os.path.join(output_dir, "raw")
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cropping_enabled = None not in (lat_min, lat_max, lon_min, lon_max)
    dates_str = generate_daily_dates(start_date, end_date)

    concat_files = []
    for date_str in dates_str:
        url = f"{url_template}{date_str}.nc"
        filename = os.path.basename(url)
        raw_path = os.path.join(raw_dir, filename)
        cropped_path = os.path.join(cropped_dir, f"cropped_{filename}" if cropping_enabled else filename)

        if not os.path.exists(raw_path):
            download_sat_file(url, raw_path)
        else:
            _logger.info(f"File already exists: {filename}")

        try:
            with xr.open_dataset(raw_path) as ds:
                ds.load()
                processed = crop_data(ds,
                                    lat_min,
                                    lat_max,
                                    lon_min,
                                    lon_max) if cropping_enabled else ds
                processed.to_netcdf(cropped_path)
                _logger.info(f"Saved {cropped_path}")
                if concat:
                    concat_files.append(processed)

        except Exception as e:
            _logger.warning(f"Failed to process {filename}: {type(e).__name__} - {e}")

    if concat and concat_files:
        try:
            concat_ds = xr.concat(concat_files, dim='time')
            concat_ds = concat_ds.assign_coords(source=sat)
            concat_filename = f"concat_{'cropped_' if cropping_enabled else ''}{sat}_{start_date}_{end_date}.nc"
            concat_path = os.path.join(output_dir, concat_filename)
            concat_ds.to_netcdf(concat_path)
            _logger.info(f"Concatenated dataset saved to {concat_path}")
            return concat_ds
        except Exception as e:
            _logger.warning(f"Failed to concatenate datasets: {type(e).__name__} - {e}")

    # Clean raw files if requested
    if clean_raw and os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        _logger.info("Raw files removed.")

    # Clean cropped files if requested
    if clean_cropped and os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
        _logger.info("Cropped files removed.")

    return None

def main(start_date: str,
         end_date: str,
         sat_list: List,
         output_dir: Union[str, os.PathLike],
         lat_min: Optional[float] = None,
         lat_max: Optional[float] = None,
         lon_min: Optional[float] = None,
         lon_max: Optional[float] = None,
         concat: bool = True,
         clean_raw: bool = False,
         clean_cropped: bool = False) -> Optional[xr.Dataset]:
    """
    Run download and processing for multiple satellites.

    Args:
        start_date: Start date in 'YYYY-MM-DD'
        end_date: End date in 'YYYY-MM-DD'
        sat: Satellite key for URL_TEMPLATES
        output_dir: Directory to save files
        lat_min, lat_max, lon_min, lon_max: Optional cropping bounds
        concat: Save a single concatenated output
        clean_raw: Delete raw files after processing

    Returns:
        xarray.Dataset if concatenated, otherwise None
    """

    all_sat = []
    for sat in sat_list:
        concat_ds = get_sat(start_date,
                            end_date,
                            sat,
                            output_dir,
                            lat_min,
                            lat_max,
                            lon_min,
                            lon_max,
                            concat=concat,
                            clean_raw=clean_raw,
                            clean_cropped=clean_cropped)
        if concat_ds is not None:
            all_sat.append(concat_ds)

    if all_sat:
        try:
            all_sat_ds = xr.concat(all_sat, dim='time')
            all_sat_filename = f"multisat_{'cropped_' if lat_min else ''}_{start_date}_{end_date}.nc"
            all_sat_path = os.path.join(output_dir, all_sat_filename)
            concat_ds.to_netcdf(all_sat_path)
            _logger.info(f"Concatenated dataset saved to {all_sat_path}")
            return all_sat_ds
        except Exception as e:
            _logger.warning(f"Failed to concatenate all satellite datasets: {type(e).__name__} - {e}")
    else:
        _logger.warning("No satellite datasets were successfully processed.")

    return None

if __name__ == "__main__":
    main(
        start_date="2019-06-01",
        end_date="2019-11-15",
        sat_list=['sentinel3a','sentinel3b','jason2','jason3','cryosat2','saral'],
        output_dir="./sat_data",
        lat_min=49.109,
        lat_max=66.304309,
        lon_min=156.6854,
        lon_max=-156.864,
        concat=True,
        clean_raw=False,
        clean_cropped=False
    )