from pathlib import  Path
import rasterio
import geopandas as gpd
import os
from rasterio import features
import numpy as np

# the path to our raster
#raster_path = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Project_Scotland/data/raster_file/big.tif'

# open the training polygons and assign them to a geopandas dataframe
#training_vectors_path = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/SparseLabels/data/vector_file' + '/training_vectors_final.geojson'
#training_vectors = gpd.read_file(training_vectors_path)
# the directory where to save our output polygons
#out_dir = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/mask'

def vector2raster(raster_path, gdf, out_dir, class_column='numerical_classes'):
    """Burns a geometry into an empty raster using rasterio"""
    raster_path = Path(raster_path)
    out_dir = Path(out_dir)
    out_path = out_dir/f"{raster_path.stem}.tif"
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        polys = ((r.geometry, r[class_column]) for _,r in gdf.to_crs(src.crs).iterrows())

        mask = features.rasterize(polys, transform=src.transform, out_shape=src.shape, all_touched=True,fill=-999)
        assert len(np.unique(mask) > 1), "No values returned"
        meta = src.meta.copy()
        meta.update(nodata=-999, count=1, dtype=rasterio.float32)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mask, indexes=1)


# rasterize our training polygons
#vector2raster(raster_path=raster_path,gdf=training_vectors,out_dir=out_dir)

