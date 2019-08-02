#%%
from pathlib import Path
from matplotlib import pyplot as plt

# 
import os
import sys
from rss_general_classifier.classifier import classic_classifier, multi_band_classifier, plot_confusion_matrix, sample_vectors_to_raster, dataframe_to_features
import gdal
import numpy as np

import xarray as xr
import xgeo # Needs to be imported to use geo extension


#%%
# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path().resolve()
print(BASE_DIR)

Prj_Path = BASE_DIR
print(Prj_Path)

# Prj_Path = os.chdir(os.path.join(Shared, './rss_general_classifier/rss_general_classifier')
raster_data_path = Prj_Path.joinpath('./crop_p224r63_all_bands.nc') # os.path.join(Prj_Path, file_path)

train_data_path = Prj_Path.joinpath('./train_data_29192.shp')

print(raster_data_path)


raster_data_path
raster_dataset = xr.open_dataset(raster_data_path, decode_times=False, chunks={'x': 100, 'y': 100})


import geopandas as gpd
shp = gpd.read_file(train_data_path)

# This line to cast the string in the shape file to int so xgeo would not give an error 
shp["class_id"] = shp["class_id"].astype(int)

# Sample the dataset with the given vector file
raster_dataset.geo.add_mask(vector_file=shp, value_name="class_id")
sampled_labeled_raster = raster_dataset.geo.sample(vector_file=shp, value_name='class_id')
raster_dataset2