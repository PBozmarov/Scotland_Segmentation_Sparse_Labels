import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def training_testing_split(df,test_size=0.25):
        '''
        input  -  a dataframe
        output -  train,test datasets. 
        
        This function is more useful than the sklearn's train_test_split, because we make sure
        that if we have enough initial data for each class in our dataframe, then we will have
        enough testing samples from each class.
        
        '''

        # empty lists in which we will append the train,test dataset for each class
        # in the dataframe
        train_to_merge=[]
        test_to_merge=[]
        
        # loop through the classes of our initial dataframe
        for klas in np.unique(df.classes):

                # dataframe that consists of all samples that lie in the class: klass
                # in our dataframe df
                dataset = df[df['classes']==klas]
                
                # we make train,test split in dataset
                train,test = train_test_split(dataset,test_size=test_size)

                # append the train,test pieces of dataset from above
                train_to_merge.append(train)
                test_to_merge.append(test)

        # concatenate by rows all the 'small' training and testing datasets
        train_set_vectors = pd.concat(train_to_merge,axis=0)
        test_set_vectors = pd.concat(test_to_merge,axis=0)

        return train_set_vectors,test_set_vectors

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window

def rasterize_vectorset(vectorset,raster_file_path,class_dict):
    
    '''
    This function labels each pixel from the raster img that lies in the geometry of the dataset

    input: vectorset - the set of our vectors
           raster_file_path - the path to our rasterfile
           class_dict - dictionary with our classes

    '''

    # a custom function for getting each value from the raster
    def all_values(x):
        return x

    # this larger cell reads data from a raster file for each training vector
    X_raw = []
    y_raw = []
    polygons_ids=[] # We take each pixel from a polygon. Here we keep the index of the corresponding polygon
    with rasterio.open(raster_file_path, 'r') as src:
        vectorset=vectorset.to_crs(src.crs)
        for (label, geom, id) in tqdm(zip(vectorset.classes, vectorset.geometry, vectorset.id), total=len(vectorset)):
           
            # read the raster data matching the geometry bounds
            window = bounds_window(geom.bounds, src.transform)
            # store our window information
            window_affine = src.window_transform(window)
            fsrc = src.read(window=window)
            # rasterize the (non-buffered) geometry into the larger shape and affine
            mask = rasterize(
                [(geom, 1)],
                out_shape=fsrc.shape[1:],
                transform=window_affine,
                fill=0,
                dtype='uint8',
                all_touched=True
            ).astype(bool)
            w, h = mask.shape[:]
            if w==0 and h==0:
                continue
            # for each label pixel (places where the mask is true)
            label_pixels = np.argwhere(mask)    
            
            for (row, col) in label_pixels:
                # add a pixel of data to X
                data = fsrc[:,row,col]
                one_x = np.nan_to_num(data, nan=1e-3)
                X_raw.append(one_x)
                # add the label to y
                y_raw.append(class_dict[label])

                # add the pixel's polygon index
                polygons_ids.append(id)

    return X_raw,y_raw,polygons_ids

def convert_raster_to_3band_raster(raster_image,new_rgb_image,r,g,b,brightness=1):
    '''
    This function takes as an input a path to a raster image, the band indexes of the red, green, and blue colors. Also
    we input the destination and the name of our new rgb image. Also we have a coefficient called brightness, which adds
    brightness to the image.
    
    The function outputs only the rgb bands of the image as a new raster.

    '''   
    # load the raster image
    src = rasterio.open(raster_image, 'r')
    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        count=1,
    )
    meta = src.meta.copy()
    img_arr = src.read()

    # read the bands r,g,and b
    red_band = img_arr[r]
    green_band = img_arr[g]
    blue_band = img_arr[b]

    # our new rgb raster image
    rgb_raster = np.dstack((red_band,green_band,blue_band))

    # reshape our raster to the appropriate form
    rgb_raster = np.transpose(rgb_raster,(2,0,1))*brightness

    meta['count'] = 3

    # save our image
    with rasterio.open(new_rgb_image, 'w', **meta) as outds:
                outds.write(rgb_raster)

    src.close()

def further_train_test_split(train_data,test_data,raster_file_path,class_dict):

    '''
    Input: train and test data
    Output: X_train,y_train,X_test,y_test
    '''
    ## RASTERIZE THE TRAINING SET
    X_train_raw,y_train_raw,train_polygons_ids = rasterize_vectorset(train_data,raster_file_path,class_dict)

    # convert the training data lists into the appropriate numpy array shape and format for scikit-learn
    X_train = np.array(X_train_raw)
    y_train = np.array(y_train_raw)

    ## RASTERIZE THE TEST SET
    X_test_raw,y_test_raw,test_polygons_ids = rasterize_vectorset(test_data,raster_file_path,class_dict)

    # convert the training data lists into the appropriate numpy array shape and format for scikit-learn
    X_test = np.array(X_test_raw)
    y_test = np.array(y_test_raw)

    return X_train,y_train,X_test,y_test