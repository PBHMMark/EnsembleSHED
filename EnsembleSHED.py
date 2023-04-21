import os
import math
from math import pi
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

from shapely import geometry, ops
from shapely.ops import unary_union
from shapely.geometry import LineString
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import nearest_points

from pysheds.grid import Grid
import rasterio
from rasterio.enums import Resampling
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
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
    if not os.path.exists("output/ShedStats"):
        os.makedirs("output/ShedStats")

def get_resolution_from_raster(elevation_raster):
    with rasterio.open(elevation_raster) as src:
        # Get the x and y pixel sizes
        res = src.res
        x_res, y_res = res[0], res[1]

        # Check if x and y resolutions are the same
        if x_res != y_res:
            raise ValueError("X and Y resolutions must be the same.")

        return x_res

def calculate_threshold(resolution, area):
    cell_area = resolution ** 2  # Calculate area of each cell in square meters
    threshold_cells = area * 1000000 / cell_area  # Convert desired area from km² to square meters and divide by cell area
    return threshold_cells

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

@timer
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
def ensemble_shed(raster_dir, x, y, threshold_area, resampling_methods, resolutions):

    for file_name in os.listdir(raster_dir):
        if file_name.endswith('.tif'):  # Check that the file is a TIFF file
            elevation_raster = os.path.join(raster_dir, file_name)
            nres = get_resolution_from_raster(elevation_raster)
            native_threshold_cells = calculate_threshold(nres, threshold_area)
            print("Native Resolution: " + str(nres) + ", threshold cells: " + str(native_threshold_cells))

            catchment_path = f'output/sheds/shed_{str(int(nres))}.shp'
            plot_path = f'output/plots/plot_{str(int(nres))}.png'

            grid, dem, dirmap, fdir, acc, clipped_catch = delineate_catchment(elevation_raster, catchment_path, x, y, native_threshold_cells)
            plot_results(grid, dem, dirmap, fdir, acc, clipped_catch, plot_path)

            #loop over resampling methods and resolutions
            for method in resampling_methods:
                for res in resolutions:
                    if res > nres:
                        factor = nres / res
                        res_threshold_cells = calculate_threshold(res, threshold_area)
                        print("Method, resolution, threshold cells: "+str(method), str(int(res)), str(res_threshold_cells))
                        resampled_path = f'output/rasters/Resampled_DEM_{method}_{int(nres)}to{int(res)}.tif'  # Path for the output raster file
                        shed_path = f'output/sheds/shed_{method}_{int(nres)}to{int(res)}.shp'
                        plot_path = f'output/plots/plot_{method}_{int(nres)}to{int(res)}.png'

                        resample_raster(elevation_raster, resampled_path, factor, getattr(Resampling, method))  # Call the resample_raster() function
                        grid, dem, dirmap, fdir, acc, clipped_catch = delineate_catchment(resampled_path, shed_path, x, y, res_threshold_cells)
                        plot_results(grid, dem, dirmap, fdir, acc, clipped_catch, plot_path)

                    else:
                        print(f"Skipping resampling for resolution {res}")

    max_shed = get_max_extent()
    min_shed = get_min_extent()
    convert_polygons_to_polylines()

def calc_hausdorff_distance(first_shp, second_shp):
    # Load the two polygons as Shapely objects
    shp1 = first_shp.geometry.values[0]
    shp2 = second_shp.geometry.values[0]

    hausdorff_dist = shp1.hausdorff_distance(shp2)

    return hausdorff_dist

def jaccard_similarity(first_shp, second_shp):
    # Load the two polygons as Shapely objects
    shp1 = first_shp.geometry.values[0]
    shp2 = second_shp.geometry.values[0]

    intersection_area = shp1.intersection(shp2).area
    union_area = shp1.union(shp2).area

    jaccard_sim = intersection_area / union_area

    return jaccard_sim

def shed_stats(resolutions):
    # List of Directories
    dirs = ["output/sheds", "output/ensembleSHEDs"]

    #Identify finest resolution
    min_res = min(resolutions)
    min_shed_path = os.path.join("output/sheds", f"shed_{min_res}.shp")

    #Calculate stats for finest resolution
    min_gdf = gpd.read_file(min_shed_path)
    min_area = min_gdf.geometry.area
    min_perimeter = min_gdf.geometry.length
    min_polsby_popper = (4 * math.pi * min_area) / min_perimeter ** 2
    min_centroid = min_gdf.centroid.values[0]

    shed_stats = []
    abs_diff_stats = []

    shed_stats.append([f"shed_{min_res}.shp", min_area.item(), min_perimeter.item(),
                       min_polsby_popper.item(), min_centroid])

    for directory in dirs:
        for file_name in os.listdir(directory):
            if file_name.endswith(".shp") and os.path.join(directory, file_name) != min_shed_path:
                print(file_name)
                file_path = os.path.join(directory, file_name)
                gdf = gpd.read_file(file_path)

                area = gdf.geometry.area
                perimeter = gdf.geometry.length
                polsby_popper = (4 * math.pi * area) / perimeter ** 2
                hausdorff_dist = calc_hausdorff_distance(min_gdf, gdf)
                centroid = gdf.centroid.values[0]
                jaccard_sim = jaccard_similarity(min_gdf, gdf)

                shed_stats.append([file_name, area.item(), perimeter.item(),
                                   polsby_popper.item(), centroid])

                abs_diff_area = abs(min_area - area)
                abs_diff_perimeter = abs(min_perimeter - perimeter)
                abs_diff_polsby_popper = abs(min_polsby_popper - polsby_popper)
                abs_centroid_distance = abs(min_centroid.distance(centroid))

                abs_diff_stats.append([file_name,abs_diff_area.item(),abs_diff_perimeter.item(),
                                       abs_diff_polsby_popper.item(),hausdorff_dist,
                                       abs_centroid_distance, jaccard_sim])


    shed_stats_df = pd.DataFrame(shed_stats, columns=['file_name',
                                                          'area',
                                                          'perimeter',
                                                          'polsby_popper',
                                                          'centroid'])

    abs_diff_stats_df = pd.DataFrame(abs_diff_stats, columns=['file_name',
                                                          'area_difference',
                                                          'perimeter_difference',
                                                          'polsby_popper_difference',
                                                          'hausdorff_dist',
                                                          'centroid_difference',
                                                          'jaccard_sim'])

    # Export the DataFrame as a CSV file
    shed_stats_df.to_csv('output/ShedStats/shed_stats_table.csv', index=False)
    abs_diff_stats_df.to_csv('output/ShedStats/shed_difference_table.csv', index=False)

    return shed_stats_df, abs_diff_stats_df

def normalize_stats(abs_diff_stats_df):
    # Get the minimum and maximum values for each column
    min_values = abs_diff_stats_df.iloc[:, 1:].min()
    max_values = abs_diff_stats_df.iloc[:, 1:].max()

    # Normalize each column using min-max normalization
    normalized_df = (abs_diff_stats_df.iloc[:, 1:] - min_values) / (max_values - min_values)
    normalized_df = pd.concat([abs_diff_stats_df.iloc[:, 0], normalized_df], axis=1)

    return normalized_df

def plot_heatmap(df):
    df_to_plot = df.drop(df.columns[0], axis=1)
    y_labels = df['file_name']
    plt.figure(figsize=(10, 8)) # set figure size to 10x8 inches
    ax = sns.heatmap(df_to_plot, cmap=sns.color_palette("crest", as_cmap=True), yticklabels=y_labels, annot=True, fmt=".2f", linewidths=0.5)
    ax.tick_params(axis='y', labelsize=8) # increase font size of y-axis tick labels
    plt.savefig('output/ShedStats/heatmap.png', dpi=300, bbox_inches='tight')

def combined_score(similarity_df):
    cmap = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    similarity_df = similarity_df.set_index('file_name')
    ax = similarity_df.plot(kind='bar', stacked=True, ylim=(0, 6), color=cmap, alpha=0.75)
    ax.set_ylabel('Combined Similarity Score (0-6)')
    ax.set_xticklabels(similarity_df.index, rotation=45, ha='right')

    similarity_df['sum'] = similarity_df.sum(axis=1)
    # Add the value on top of each bar
    for i, v in enumerate(similarity_df['sum']):
        ax.text(i, v + 0.1, str(round(v, 2)), color='black', ha='center')

    plt.tight_layout()
    plt.savefig('output/ShedStats/combinedscore.png')

def create_radar_plots(df):
    columns = [col for col in df.columns if col != 'file_name']
    num_rows = df.shape[0]
    num_cols = int(
        np.ceil(np.sqrt(num_rows)))  # Calculate the number of columns based on the square root of the number of rows

    fig, axs = plt.subplots(num_cols, num_cols, figsize=(6 * num_cols, 6 * num_cols), subplot_kw=dict(polar=True))

    for i, (_, row) in enumerate(df.iterrows()):
        values = row[columns].values.tolist()
        values += values[:1]
        angles = [n / float(len(columns)) * 2 * np.pi for n in range(len(columns))]
        angles += angles[:1]

        ax = axs[i // num_cols, i % num_cols] if num_rows > 1 else axs[
            i % num_cols]  # Use a single axis if there's only one row
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), columns)
        ax.set_title(row['file_name'])

    # Remove any unused subplots
    for i in range(num_rows * num_cols, len(axs.flat)):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.savefig('output/ShedStats/radar_plots.png')

def compare_similarity(abs_diff_stats_df):
    normalized_df = normalize_stats(abs_diff_stats_df)
    df_temp = normalized_df.drop(['file_name', 'jaccard_sim'], axis=1)
    inverted_df = 1 - df_temp
    similarity_df = pd.concat([normalized_df['file_name'], inverted_df, normalized_df['jaccard_sim']], axis=1)

    similarity_df.to_csv('output/ShedStats/similarity_df.csv', index=False)
    plot_heatmap(similarity_df)
    combined_score(similarity_df)
    create_radar_plots(similarity_df)


if __name__ == '__main__':
    #FortMac
    raster_dir = "C:\PhD\Courses\GEOG825\Capstone\DEMs\FortMac"
    x, y = 689363.1474, 6311325.8758

    #Jasper
    #raster_dir = "C:\PhD\Courses\GEOG825\Capstone\DEMs\Jasper"
    #x, y = 318092.9584, 5884246.0063 # 318086.5806, 5884252.1877


    #Jasper 318086.5806, 5884252.1877
    #BowValleyTrail Near Grotto Canyon 486295.0985, 5655136.4168
    #Canmore Cougar Creek 477106.1775, 5657174.6435
    threshold_area = 1 #km^2

    resampling_methods = ['nearest', 'bilinear', 'cubic'] # Change this to the desired resampling methods
    resolutions =[1,15,30,60,90,120]

    create_output_dirs()
    #ensemble_shed(raster_dir, x, y, threshold_area, resampling_methods, resolutions)
    #shed_stats_df, abs_diff_stats_df = shed_stats(resolutions)

    abs_diff_stats_df = pd.read_csv("C:\PhD\Courses\GEOG825\Capstone\EnsembleShed\Lib\output\ShedStats\shed_difference_table.csv")
    compare_similarity(abs_diff_stats_df)




