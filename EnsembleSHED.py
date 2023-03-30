#option to burn stream? or leave to them to do so and use in import
#how to select flow acc? set and area min and calcualte it based on resoltuion? , include in ensemble?
#currently flow acc is changing base on resoltuoin

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely import geometry, ops
from shapely.ops import unary_union
from shapely.geometry import LineString
from pysheds.grid import Grid
import rasterio
from rasterio.enums import Resampling
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import colors
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def create_output_dirs():
    """Create the necessary output directories"""
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/rasters"):
        os.makedirs("output/rasters")
    if not os.path.exists("output/sheds"):
        os.makedirs("output/sheds")
    if not os.path.exists("output/plots"):
        os.makedirs("output/plots")
    if not os.path.exists("output/ensembleSHEDs"):
        os.makedirs("output/ensembleSHEDs")


def get_resolution_from_raster(elevation_raster):
    with rasterio.open(elevation_raster) as src:
        # Get the x and y pixel sizes
        res = src.res
        x_res, y_res = res[0], res[1]
        print(f"X resolution: {x_res}, Y resolution: {y_res}")

        # Check if x and y resolutions are the same
        if x_res != y_res:
            raise ValueError("X and Y resolutions must be the same.")

        return x_res


def resample_raster(input_path, output_path, upscale_factor, method):
    with rasterio.open(input_path) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=method
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        # create output raster file
        output_profile = dataset.profile.copy()
        output_profile.update(
            width=data.shape[-1],
            height=data.shape[-2],
            transform=transform,
            dtype=data.dtype
        )

        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(data)


def delineate_catchment(elevation_raster,shed_path, x, y, threshold):
    # Read the input raster object
    grid = Grid.from_raster(elevation_raster)
    dem = grid.read_raster(elevation_raster)

    # Condition DEM
    pit_filled_dem = grid.fill_pits(dem)  # Fill pits in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)  # Fill depressions in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)  # Resolve flats in DEM

    # Compute flow directions
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # Specify D8 directional mapping
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)  # Compute flow directions

    # Calculate flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)  # Calculate flow accumulation

    # Delineate a catchment
    x_snap, y_snap = grid.snap_to_mask(acc > threshold, (x, y))  # Snap pour point to high accumulation cell
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')

    # Clip the bounding box to the catchment
    grid.clip_to(catch)
    clipped_catch = grid.view(catch)

    shapes = grid.polygonize()
    catchment_polygon = ops.unary_union([geometry.shape(shape) for shape, value in shapes])

    # Create a GeoDataFrame from the polygon
    catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_polygon])
    catchment_gdf = catchment_gdf.set_crs(epsg=3400)

    # Write the GeoDataFrame to a shapefile
    catchment_gdf.to_file(shed_path, driver='ESRI Shapefile')

    return grid, dem, dirmap, fdir, acc, clipped_catch


def plot_results(grid, dem, dirmap, fdir, acc, clipped_catch, plot_path):
    # Create figure with 4 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.patch.set_alpha(0)

    # Plot DEM
    axs[0, 0].imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
    axs[0, 0].set_title('Digital elevation map', size=14)
    axs[0, 0].set_xlabel('Longitude')
    axs[0, 0].set_ylabel('Latitude')
    axs[0, 0].grid(zorder=0)
    axs[0, 0].set_aspect('equal')

    fig.colorbar(axs[0, 0].imshow(dem, extent=grid.extent, cmap='terrain', vmax=dem.max(), vmin=dem.min()),
                 ax=axs[0, 0])

    # Plot flow direction grid
    boundaries = ([0] + sorted(list(dirmap)))
    axs[0, 1].imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
    axs[0, 1].set_title('Flow direction grid', size=14)
    axs[0, 1].set_xlabel('Longitude')
    axs[0, 1].set_ylabel('Latitude')
    axs[0, 1].grid(zorder=-1)
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_ylim(axs[0, 1].get_ylim()[::-1]) # Invert y-axis

    fig.colorbar(axs[0, 1].imshow(fdir, extent=grid.extent, cmap='viridis'), ax=axs[0, 1], boundaries=boundaries,
                 values=sorted(dirmap))

    # Plot flow accumulation
    im = axs[1, 0].imshow(acc, extent=grid.extent, zorder=2,
                          cmap='cubehelix',
                          norm=colors.LogNorm(1, acc.max()),
                          interpolation='bilinear')
    axs[1, 0].set_title('Flow accumulation', size=14)
    axs[1, 0].set_xlabel('Longitude')
    axs[1, 0].set_ylabel('Latitude')
    axs[1, 0].grid(zorder=0)
    axs[1, 0].set_aspect('equal')
    fig.colorbar(im, ax=axs[1, 0], label='Upstream Cells')

    # Plot delineated catchment
    axs[1, 1].imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
                      zorder=1, cmap='Greys_r')
    axs[1, 1].set_title('Delineated catchment', size=14)
    axs[1, 1].set_xlabel('Longitude')
    axs[1, 1].set_ylabel('Latitude')
    axs[1, 1].set_aspect('equal')
    axs[1, 1].grid(zorder=0)

    # Adjust subplot spacing and save figure
    fig.tight_layout()
    plt.savefig(plot_path)

def get_max_extent():
    # Get all shapefiles in the directory
    directory = "output/sheds"
    max_shed = "output/ensembleSHEDs/max_shed.shp"

    shp_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.shp')]

    # Merge shapefiles into a single GeoDataFrame
    gdf_list = []
    for shp_file in shp_files:
        gdf = gpd.read_file(shp_file)
        gdf_list.append(gdf)

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf.crs)

    # Dissolve the merged GeoDataFrame
    dissolved_gdf = merged_gdf.dissolve(by='FID')

    # Save the dissolved shapefile
    dissolved_gdf.to_file(max_shed)

    return dissolved_gdf

def get_min_extent():
    directory = "output/sheds"

    # Read all polygons from directory into a GeoDataFrame
    polygons = gpd.read_file(directory)

    # Compute the intersection of all polygons
    intersection = unary_union(polygons.geometry)

    # Create a new GeoDataFrame with the intersection polygon
    intersection_df = gpd.GeoDataFrame(geometry=[intersection])

    min_extent_path = 'output/ensembleSHEDs/min_shed.shp'

    crs = polygons.crs

    # Write the intersection polygon to a shapefile with the same projection as the input shapefiles
    intersection_df.to_file(min_extent_path, driver='ESRI Shapefile', crs=crs)

def convert_polygons_to_polylines():
    input_directory = "output/sheds"
    output_directory = "output/lines"
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize an empty list to store the polylines
    polyline_list = []

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.shp'):
            # Read the polygons from the input file
            filepath = os.path.join(input_directory, filename)
            polygons = gpd.read_file(filepath)

            # Set the geometry column
            polygons = polygons.set_geometry('geometry')

            # Convert polygons to polylines
            polylines = polygons.boundary

            # Set the CRS if it is not already set
            if polylines.crs is None and polygons.crs is not None:
                polylines.crs = polygons.crs

            # Append the polylines to the list
            polyline_list.append(polylines)

            # Save the polylines to a new shapefile in the output directory
            output_filename = os.path.splitext(filename)[0] + '_polyline.shp'
            output_filepath = os.path.join(output_directory, output_filename)
            polylines.to_file(output_filepath)

    # Merge all the polylines together
    merged_polylines = unary_union(polyline_list)

    # Create a GeoDataFrame from the merged polylines
    merged_polylines_gdf = gpd.GeoDataFrame(geometry=[merged_polylines])

    # Set the CRS to the CRS of the first input polygon
    first_filename = [f for f in os.listdir(input_directory) if f.endswith('.shp')][0]
    first_filepath = os.path.join(input_directory, first_filename)
    first_polygon = gpd.read_file(first_filepath)
    if first_polygon.crs is not None:
        merged_polylines_gdf.crs = first_polygon.crs

    # Save the merged polylines to a new shapefile in the output directory
    output_filepath = os.path.join(output_directory, 'merged_polylines.shp')
    merged_polylines_gdf.to_file(output_filepath)


@timer
def ensemble_shed(elevation_raster, x, y, threshold, resampling_methods, nres, resolutions):

    #perform delienation for native resoluiton
    catchment_path = 'output/sheds/shed_original.shp'
    plot_path = 'output/plots/plot_original.png'
    grid, dem, dirmap, fdir, acc, clipped_catch = delineate_catchment(elevation_raster, catchment_path, x, y, threshold)
    plot_results(grid, dem, dirmap, fdir, acc, clipped_catch, plot_path)

    #loop over resampling methods and resolutions
    for method in resampling_methods:
        for res in resolutions:
            if res != nres:
                factor = nres / res
                print(factor, method, res)
                resampled_path = f'output/rasters/Resampled_DEM_{method}_{res}.tif'  # Path for the output raster file
                shed_path = f'output/sheds/shed_{method}_{res}.shp'
                plot_path = f'output/plots/plot_{method}_{res}.png'

                resample_raster(elevation_raster, resampled_path, factor, getattr(Resampling, method))  # Call the resample_raster() function
                grid, dem, dirmap, fdir, acc, clipped_catch = delineate_catchment(resampled_path, shed_path, x, y, threshold)
                plot_results(grid, dem, dirmap, fdir, acc, clipped_catch, plot_path)
            else:
                print(f"Skipping resampling for resolution {res}")

    max_shed = get_max_extent()
    min_shed = get_min_extent()
    convert_polygons_to_polylines()


if __name__ == '__main__':
    elevation_raster = "C:/PhD/Courses/GEOG825/Capstone/Dev/Data/Canmore_DEM_MERIT_proj_Resam1.tif"
    x, y = 477106.1775, 5657174.6435 #-115.327, 51.086   # Specify pour point
    threshold = 300

    resampling_methods = ['nearest', 'bilinear', 'cubic'] # Change this to the desired resampling methods
    resolutions =[120,300]

    create_output_dirs()
    nres = get_resolution_from_raster(elevation_raster)

    ensemble_shed(elevation_raster, x, y, threshold, resampling_methods, nres, resolutions)




