import os
from itertools import product
import rasterio as rio
from rasterio import windows
import argparse
from pathlib import Path
from tqdm import tqdm 

"""
Takes care of the butchery to convert the big disgraces into nice tiles for our deep learning framework to ingest
Test if this is faster to read, rather than window reading with gdal? (which I suspect it is, easier for mixing up and stuff)
"""

def find_name(path):
   '''
   Input: path - path to a folder containing a single tif image file.
   Output: the name of the image file
   ''' 
   
   # we split the path by /
   parts = path.split('/')
   
   # the name is the last element of the list parts but without the .tif extension
   name = parts[-1][:-4]
   
   return name


def _get_tiles(ds, width=256, height=256, discard_truncated=True):
    """Tiling a big raster for deep learning

    Arguments: \n
    `ds` (`rasterio raster`): Source raster to get the tiles from \n
    `width` (`int`): Width (in pixels) of each tile (default=256)\n
    `height` (`int`): Height (in pixels) of each tile (default=256) \n
    `discard_truncated` (`bool`): Discard resulting tiles with width or height smaller than specified (default=True)
    """
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in  offsets:
        # if row_off != 3712 and col_off != 5632 : 
            #getting rid of the edges (smaller than 128*128) specific to width/height!!!!
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        if discard_truncated:
            if window.width == width and window.height == height:
                yield window, transform
            else: pass
        else:
            yield window, transform

def tile_raster(input_path, output_folder, width=256, height=256, discard_truncated=True):
    """Writes tiles generated from get_tiles to the output folder"""
    name = find_name(input_path) # get the name of the raster file in a pesky, unreadable way
    output_filename = '{}_tile_{}-{}.tif'
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    with rio.open(input_path) as inds:
        meta = inds.meta.copy()

        for window, transform in _get_tiles(inds, width=width, height=height, 
                                           discard_truncated = discard_truncated):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(output_folder,output_filename.format(name, int(window.col_off), int(window.row_off)))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))


# MAKE TILES FROM THE ORIGINAL RASTER ( ORIGINAL TIF IMAGE )

# we first get the tiles for the original tif_image
#input_path = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/original_tif_image/Trans_nzoia_2019_05-02.tif'
#output_folder = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/original_tif_image/tiled'

# MAKE TILES
#tile_raster(input_path,output_folder)


# MAKE TILES FROM THE MASK FILE TIF IMAGE
#input_path = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/mask/training_vectors_rasterized_python.tif'
#output_folder =  '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/mask/tiled'

#tile_raster(input_path,output_folder)