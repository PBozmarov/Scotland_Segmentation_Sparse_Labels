import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image


#X_dir = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/original_tif_image/tiled'
#X_names = os.listdir(X_dir)

#y_dir = '/Users/pavelbozmarov/Desktop/Desktop/SPACE INTELLIGENCE/Random_Forests/TIF DATA/mask/tiled'
#y_names = os.listdir(y_dir)


def create_dataset_XY(X_names,X_dir,y_dir):
    '''
    We run through the whole dataset of names X_names, and we append each image 
    to a X set and each corresponding label
    to y set
    '''
    
    X=[] # images
    y=[] # labels
    
    # loop through the image names in X_names
    for image_name in X_names:
        
        # find the corresponding label for each image
        label_name = find_label(image_name)

        # the path to the image
        image_path = X_dir + '/' + image_name
        
        # find the label path
        label_path = y_dir + '/' + label_name

        # read the image and the label
        image = rasterio.open(image_path)
        label = rasterio.open(label_path)

        # convert the image and the label to numpy array
        image_arr = image.read()
        label_arr = label.read()

        X.append(image_arr) # add the array to our X list
        y.append(label_arr) # add the label to our y list
    
    return (X,y)

def find_label(image_name):
    '''
    Function that takes the name of an image folder in landsat8 and returns
    the appropriate label folder name
    '''
    #split the image_name by _ symbol.
    parts = image_name.split('_')
    
    # we take the last part
    last_part = parts[-1]
    
    label_name = 'rasterized_vectors_tile_' + last_part
    #return the name of the label folder
    return label_name

def clear_XY(X,y):
    '''
    Input: X set and y set. 
    Output: We clear the nans from X ( and the corresponding y values) and also clear y
            from empty labels

    '''
    # declare the clear lists
    clear_X = []
    clear_y = []

    # loop through the items of y
    for X_item,y_item in zip(X,y):
        
        # if the maximum is larger or equal than 0
        # this means that the label and the corresponding X item are clear
        # clear item means that it is not full of nans - it has at least one non nan value
        
        if X_item.max() >= 0 and y_item.max() >=0 :

            # they are clear so we append them
            clear_X.append(X_item)
            clear_y.append(y_item)

    
    return clear_X,clear_y

def display_rasters(img_name,X_dir,y_dir,brightness = 6):
    '''
    input: img_name - the name of the tile image from the original raster image
           brightness - tunning coefficient related to the plotting brightness
           of the image
    '''
    # we find the label name
    label_name = find_label(img_name)
    
    # the path to the image
    image_path = X_dir + '/' + img_name
    
    # find the label path
    label_path = y_dir + '/' + label_name

    # read the image and the label
    image = rasterio.open(image_path)
    label = rasterio.open(label_path)

    # convert the image and the label to numpy array
    image_arr = image.read()
    label_arr = label.read()

    # Create numpy arrays from the r,g, and b pictures.
    red_arr = image_arr[2]
    blue_arr = image_arr[0]
    green_arr = image_arr[1]
    
    # Stack the r,g,b arrays and apply brightness
    #rgb_arr = (np.dstack((red_arr,green_arr,blue_arr)) * 256) .astype(np.uint8) * brightness
    rgb_arr = image_arr[:3][::-1]*brightness
    rgb_arr = rgb_arr.transpose(1,2,0)
    #PLOTTING
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.set_title('IMAGE',fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(rgb_arr)

    ax2.set_title('LABEL',fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(label_arr[0],aspect="auto")
    plt.colorbar(ax2.imshow(label_arr[0]),fraction=0.046, pad=0.04)
    plt.show();

   
    print('0 - Agriculture, 1 - Bares_and_Built, 2 - Bogs ')
    print('3 - Forest, 4 - Grassland, 5 - Shrub/Heathland')
    print('6 - Water, 9999 - No Data')

def plt_image(i,X):# check if everything is ok
    # Create numpy arrays from the r,g, and b pictures.
    image_arr = X[i]
    red_arr = image_arr[2]
    blue_arr = image_arr[0]
    green_arr = image_arr[1]
        
    # Stack the r,g,b arrays and apply brightness
    #rgb_arr = (np.dstack((red_arr,green_arr,blue_arr)) * 256) .astype(np.uint8) * brightness
    rgb_arr = image_arr[:3][::-1]*5
    rgb_arr = rgb_arr.transpose(1,2,0)
    plt.imshow(rgb_arr)
    plt.show()

def plt_label(i,y):

    label_arr = y[i]
    plt.imshow(label_arr[0])
    plt.show()



def display(index,brightness=4):

    '''
    This function gets an index number and displays the picture with this index number and its'
    corresponding label. Index corresponds to the order of the picture subfolders inside our main 
    folder landsat8 brightness - 0 means no added brightness to the original picture. 
    With 1,2,3,.... we increase the brightness of the picture.
    '''
    index = index
    # Paths to red, green, and blue layers of our satelite picture
    red_path = f'{landsat_dir}/{images[index]}/B04.tif'
    green_path = f'{landsat_dir}/{images[index]}/B03.tif'
    blue_path = f'{landsat_dir}/{images[index]}/B02.tif'
    
    # Create r,g, and b pictures based on the pictures'paths
    red = Image.open(red_path)
    green = Image.open(green_path)
    blue = Image.open(blue_path)

    # Create numpy arrays from the r,g, and b pictures.
    red_arr = np.array(red)
    blue_arr = np.array(blue)
    green_arr = np.array(green)
        
    # Stack r,g,b layers to create a rgb pictures. Multiply by 256 to put the pictures'
    # values from 0-1 -> 0-256
    rgb_uint8 = (np.dstack((red_arr,green_arr,blue_arr)) * 256) .astype(np.uint8) 

    # Increase the brightness of the pictures
    rgb_uint8 = rgb_uint8*brightness
    image_arr = rgb_uint8
    plt.figure(figsize=(20,10))
    #plt.figsize((20,10))
   
    #finding the name of the image with the corresponding index
    image_name = images[index]
    #finding the name of the label that corresponds to that image
    label_name = find_label(image_name)
    #finding the path of this label 
    label_path = f'{landsat_labels_dir}/{label_name}/labels.tif'
    #the label image(mask)
    label_image = rasterio.open(label_path)
    #label image as array
    label_image_arr = label_image.read(1)
    
    #PLOTTING
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.set_title('IMAGE',fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(rgb_uint8)

    ax2.set_title('LABEL',fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(label_image_arr,aspect="auto")
    plt.colorbar(ax2.imshow(label_image_arr),fraction=0.046, pad=0.04)
    plt.show();
    print('0 - No Data, 1 - Water, 2 - Artificial Bare Ground, 3 - Natural Bare Ground ')
    print('4 - Snow/Ice, 5 - Woody, 6 - Cultivated Non-Woody, 7 - Natural Non-Woody ')

    return image_arr,label_image_arr