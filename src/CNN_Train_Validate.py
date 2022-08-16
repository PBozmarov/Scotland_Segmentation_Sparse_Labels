from os import path as op
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from rasterio.windows import Window
import rasterio
import pickle
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import itertools
from torchvision import datasets,transforms,models
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import csv

from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 

def create_train_test_split(X,y,test_size=0.3,random_state=42,gpu=True):
   '''
   This function creates an SKlearn traditional train_test split.
   We use this to say if we want to save our data to gpu or to cpu
   input: X,y - our data
          test_size,random_state,
          gpu - could be True/False
   output: train_data,test_data
   '''
   
   # we do train-test split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   if gpu == True:
      
      # Convert X_train, y_train, X_test, y_test to tensors and save them in gpu
      X_train = torch.FloatTensor(X_train).cuda()
      y_train = torch.LongTensor(y_train).cuda()

      X_test = torch.FloatTensor(X_test).cuda()
      y_test = torch.LongTensor(y_test).cuda()
   else:

      # Convert X_train, y_train, X_test, y_test to tensors and save them in cpu
      X_train = torch.FloatTensor(X_train)
      y_train = torch.LongTensor(y_train)

      X_test = torch.FloatTensor(X_test)
      y_test = torch.LongTensor(y_test)
      
  # Prepare the training set for batch sampling
   train_data = list(zip(X_train,y_train))

  # Prepare the testing set for batch sampling
   test_data = list(zip(X_test,y_test))

   return train_data,test_data


def augment_dataset(data,by_product=3):
    '''
    This functions expands a dataset.
    input: data, by_product - add <new_images_product * len(data)> images
    '''
    data = data.copy() # we copy the data
    
    new_images = [] # the new images to be added
    
    n = len(data) # the length of our data
    
    seeds = np.arange(by_product*n) # seeds 0,1,2,...,n*by_product

    transform_data = transforms.Compose([                                 
        
        transforms.RandomHorizontalFlip(),   # random horizontal flipping 
        transforms.RandomVerticalFlip(),   # random vertical flipping
        transforms.RandomRotation(50),  # rotate up to +/- 50 degrees 
                                              ])
       
    for i in range(by_product):  # loop our dataset <new_images_product> times
       
       for index,(x,y) in enumerate(data):
              
              manual_seed = seeds[index] # we choose our seed

              torch.manual_seed(manual_seed) # we fix our seed so the transformations we apply on X and y be the same
              new_x = transform_data(x) # apply transformation on  x
              torch.manual_seed(manual_seed) # we fix our seed so the transformations we apply on X and y be the same
              new_y = transform_data(y) #          -||-            y                
              
              # add our new item
              new_images.append( (new_x,new_y) )
    
    # add the list of our new images
    data.extend(new_images)
    
    return data


def select_class(train_data,klass):

      '''
      input:train_data , klass - the class that we want to extract
      output: we select these tuples from the training data which belong to a specific class
      '''

      result = [] 

      for (x,y) in train_data: # we loop through the itesm of train_data

          if klass in torch.unique(y): # we check whether the class is in the unique items of y

              result.append( (x,y) ) # if yes, we append (x,y) to our result

      return result


class UNet(nn.Module):




    def __init__(self, in_channels=13, out_channels=7, init_features=32):

        super(UNet, self).__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)




        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")




        self.upconv4 = nn.ConvTranspose2d(

            features * 16, features * 8, kernel_size=2, stride=2

        )

        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(

            features * 8, features * 4, kernel_size=2, stride=2

        )

        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(

            features * 4, features * 2, kernel_size=2, stride=2

        )

        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(

            features * 2, features, kernel_size=2, stride=2

        )

        self.decoder1 = UNet._block(features * 2, features, name="dec1")




        self.conv = nn.Conv2d(

            in_channels=features, out_channels=out_channels, kernel_size=1

        )




    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))




        bottleneck = self.bottleneck(self.pool4(enc4))




        dec4 = self.upconv4(bottleneck)

        dec4 = torch.cat((dec4, enc4), dim=1)

        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)

        dec3 = torch.cat((dec3, enc3), dim=1)

        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)

        dec2 = torch.cat((dec2, enc2), dim=1)

        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)

        dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        #return F.softmax(self.conv(dec1),dim=1)
        return self.conv(dec1)




    @staticmethod

    def _block(in_channels, features, name):

        return nn.Sequential(

            OrderedDict(

                [

                    (

                        name + "conv1",

                        nn.Conv2d(

                            in_channels=in_channels,

                            out_channels=features,

                            kernel_size=3,

                            padding=1,

                            bias=False,

                        ),

                    ),

                    (name + "norm1", nn.BatchNorm2d(num_features=features)),

                    (name + "relu1", nn.ReLU(inplace=True)),

                    (

                        name + "conv2",

                        nn.Conv2d(

                            in_channels=features,

                            out_channels=features,

                            kernel_size=3,

                            padding=1,

                            bias=False,

                        ),

                    ),

                    (name + "norm2", nn.BatchNorm2d(num_features=features)),

                    (name + "relu2", nn.ReLU(inplace=True)),

                ]

            )

        )

def mask_trainings_and_predictions(x_train,y_train,model):


  '''
   Input: batch from our training dataset and our model.
   Output: modified y_batch and modified predictions.
   
   Description: We work with sparse labels so most of the pixels in the elements of y_batch
                will be nodata (less than 0 ). We don't want our loss to take them into account,
                so we need to create a mask which restricts the domain of the loss function
                only on the non-negative pixel values of the elements from y_batch.

  '''

  # we find the predictions
  y_pred  = model(x_train)

  # we create the desired mask and apply it to y_train
  mask_train = (y_train >= 0)
  y_train = y_train[mask_train]

  # we reshape our mask ( remove 1 dimension )
  mask_train = mask_train.reshape([len(x_train),256,256])

  # this is our masked y_pred
  y_pred = torch.cat([y_pred[:,0,:,:][mask_train],  y_pred[:,1,:,:][mask_train],y_pred[:,2,:,:][mask_train],y_pred[:,3,:,:][mask_train],
           y_pred[:,4,:,:][mask_train],y_pred[:,5,:,:][mask_train], y_pred[:,6,:,:][mask_train]],dim=0)
  
                      
  # we fix y_pred to the appropriate shape
  y_pred = torch.transpose(y_pred.view(7,-1),0,1)
  
  return (y_train,y_pred)

def mask_trainings_and_predictions_equal_polygons(x_train,y_train,model):

 '''
   Input: batch from our training dataset and our model.
   Output: modified y_train and modified predictions.
   
   Description: We work with sparse labels so most of the pixels in the elements of y_batch
                will be nodata (less than 0 ). We don't want our loss to take them into account,
                so we need to create a mask which restricts the domain of the loss function
                only on the non-negative pixel values of the elements from y_batch. In this fun-
                ction we also consider each polygon to be equally important, no matter how small
                it is.

  '''
# record the batch size
 batch_size = y_train.shape[0]

  # we find the predictions
 y_pred  = model(x_train)

 # copy(clone) y_train
 y_train_copy = y_train.clone()

# we create the desired mask and apply it to y_train
 mask_train = (y_train >= 0)
 y_train_masked = y_train[mask_train]

 # we reshape it by removing a band ( band 1 )
 y_train_copy = y_train_copy.reshape(batch_size,256,256)

 # we merge the labels from each batch to one long image
 y_train_copy = y_train_copy.reshape(256*batch_size,256).T

 # -999 -> 0, -1->0, 0->500. 
 mask_999 = (y_train_copy==-999)
 mask_0 = (y_train_copy==0)
 mask_1 = (y_train_copy==-1)

 y_train_copy[mask_999]=0
 y_train_copy[mask_1]=0
 y_train_copy[mask_0]=500

 # we will need this structure for the following function
 s = ndimage.generate_binary_structure(2,2)

 # this is our polygons id mask
 
 polygons_id_mask = ndimage.label(y_train_copy.cpu(),structure=s)[0]
 polygons_id_mask = torch.LongTensor(polygons_id_mask.T)
 polygons_id_mask = polygons_id_mask.view(batch_size,1,256,256)

 # we use the mask_train to take only the non nan valus
 polygons_id_mask_masked = polygons_id_mask[mask_train]

 # we reshape our mask ( remove 1 dimension )
 mask_train = mask_train.reshape([len(x_train),256,256])

# this is our masked y_pred
 y_pred = torch.cat([y_pred[:,0,:,:][mask_train],  y_pred[:,1,:,:][mask_train],y_pred[:,2,:,:][mask_train],y_pred[:,3,:,:][mask_train],
           y_pred[:,4,:,:][mask_train],y_pred[:,5,:,:][mask_train], y_pred[:,6,:,:][mask_train]],dim=0)
  
                      
  # we fix y_pred to the appropriate shape
 y_pred = torch.transpose(y_pred.view(7,-1),0,1)

 # we take the indexes of the sorted polygons_id_mask_masked
 sorted_indexes_polygons_id_mask_masked = polygons_id_mask_masked.argsort()

 # we sort y_train, y_pred, and polygons_id_mask_masked based on the above indexes
 y_train_masked_sorted = y_train_masked[sorted_indexes_polygons_id_mask_masked]
 sorted_polygons_id_mask_masked = polygons_id_mask_masked[sorted_indexes_polygons_id_mask_masked]
 y_pred_masked_sorted = y_pred[sorted_indexes_polygons_id_mask_masked]


 # now we divide our y_train and y_pred into subsets for every polygon
 y_train_masked_split_by_polygons = np.split(y_train_masked_sorted, np.unique(sorted_polygons_id_mask_masked, return_index=True)[1][1:])
 y_pred_masked_split_by_polygons = np.split(y_pred_masked_sorted, np.unique(sorted_polygons_id_mask_masked, return_index=True)[1][1:])

 return y_train_masked_split_by_polygons,y_pred_masked_split_by_polygons


  # Checking our predictions
def nn_generate_predictions(y_pred):
    '''
    input: y_pred - vector with probabilities for each possible class
    output: list(for each element of y_pred we generate the predicted class based on the probabilities)
    '''

    preds_ann = [] # empty list for our predictions

    n_rows = len(y_pred) # number of rows in our y_pred
            
    for i in range(n_rows): # looping through the rows of X_test
                
          preds_ann.append( y_pred[i].argmax().item() ) # Append the prediction.
            
    return preds_ann

def plot_cm(cm,classes):
  '''
    plot the confusion matrix
    input: cm - confusion matrix
           classes - list of our classes
  '''
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  fig, ax = plt.subplots(figsize=(10, 10))
  im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  ax.figure.colorbar(im, ax=ax)

  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
      # ... and label them with the respective list entries
      xticklabels=classes, yticklabels=classes,
      title='Normalized Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
  fmt = '.2f'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
             ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  plt.show()


def train_validate_model(train_data,test_data,labels,classes,input_model = False,init_features=32,batch_size=16,epochs=20,init_lr=0.001,verbose=True,gpu=True):
    
    '''
    Function that trains our neural network
    input: train and test data, labels=[0,1,2,3,...],classes=['Agriculture',..] 
           init_features, batch_size, number of epochs,
           and gpu = True/False - whether we use gpu or not
           init_lr - initial learning rate

    output: our trained model, all true classes from the whole test set, and all
            predicted classes from the test set
    '''
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    
    # trackers for printing
    train_losses=[]
    val_losses=[]
    Epochs = list(torch.arange(epochs))

    # we check for input model
    if input_model==False:  
      
      # instatiate our model
      if gpu == True:
          model = UNet(in_channels=6,out_channels=5,init_features=init_features).cuda()
      else:
          model = UNet(in_channels=6,out_channels=5,init_features=init_features)
    
    else:
      model = input_model

    # creating our train and test loaders
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=45,shuffle=False)

    # setting our loss function and our optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(),lr = init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = 15,verbose=True)

    for i in range(epochs):
        i+=1
        
        ########## TRAINING ########
        for batch,(x_train_batch,y_train_batch) in enumerate(train_loader):
            batch+=1

            # apply mask on our y_train_batch and y_pred_batch 
            y_train_batch,y_pred_batch = mask_trainings_and_predictions(x_train_batch,y_train_batch,model)
        
            # calculate the loss for the current batch
            train_loss=criterion(y_pred_batch,y_train_batch)
            
            optimizer.zero_grad() # optimizing
            train_loss.backward()       # back propagation
            optimizer.step()      # step update

        # update the train losses
        train_losses.append(train_loss)

        if verbose == True:
            print(f"Epoch:{i}, train_loss:{train_loss}, {optimizer.param_groups[0]['lr']}")

        ######## VALIDATION ######

        # if we are in the last epoch
        if i == epochs-1:
          
          test_predictions=[] # we will append here the predictions from 
                              # y_pred_batch from each test batch
                            
          test_true = [] # we will append here the y_test_batch from each
                         # test batch

        for batch,(x_test_batch,y_test_batch) in enumerate(test_loader):

            with torch.no_grad():
                 
                 # apply mask on our y_train_batch and y_pred_batch 
                y_test_batch,y_pred_batch = mask_trainings_and_predictions(x_test_batch,y_test_batch,model)
            
            if i == epochs-1:
              # generate batch predictions from y_pred_batch
              batch_predictions = nn_generate_predictions(y_pred_batch)
              test_predictions+=batch_predictions

              # append the results for 
              test_true+=list(y_test_batch.cpu().detach().numpy())
            
            # calculate the loss for the current batch
            val_loss=criterion(y_pred_batch,y_test_batch)  

        # Update the val losses
        val_losses.append(val_loss)

        # Adjust the learning rate based on the validation loss
        scheduler.step(val_loss)
        
        if verbose == True:
            print(f"Epoch:{i}, val_loss:{val_loss}, {optimizer.param_groups[0]['lr']}")

    print(' ')
    print(f'init features: {init_features}, batch size: {batch_size}, learning rate: {init_lr}')

    ####################### PLOTTING #############################

    # plot a confusion matrix
    cm = confusion_matrix(test_true,test_predictions,labels=labels)
    plot_cm(cm,classes)


    # Plot train and validation losses
    train_losses = [item.cpu().detach() for item in train_losses]
    val_losses = [item.cpu().detach() for item in val_losses]
    plt.title('TRAIN AND VALIDATION LOSSES')
    plt.plot(Epochs,train_losses,label='train_loss',color='blue')
    plt.plot(Epochs,val_losses,label='val_loss',color='red')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model,test_true,test_predictions  

def test_statistics(test_true,test_predictions,verbose=True):

    '''
    This functions calculates accuracy,balanced_accuracy,f1,recall, and precision.
    input: test_true - the real predicted test values taken from the whole test dataloader,
           test_predictions - predicted values on the whole test dataloader.
    output: results for the statistics
    '''
     
    # the number of samples in our test loader
    n_test = len(list(test_true))

    # calculate the statistics
    bal_acc = balanced_accuracy_score(test_true,test_predictions)
    
    recall = recall_score(test_true,test_predictions,average='weighted')

    precision = precision_score(test_true,test_predictions,average='weighted')

    f1 = f1_score(test_true,test_predictions,average='weighted')
    
    accuracy = accuracy_score(test_true,test_predictions)


    
    if verbose==True:

        print(f'Accuracy:{accuracy}')
        print(f'Balanced Accuracy:{bal_acc}')
        print(f'Recall:{recall}')
        print(f'Precision:{precision}')
        print(f'f1 score:{f1}')


    return bal_acc,recall,precision,f1,accuracy

           

def export_to_csv(table_name,init_features,batch_size,test_balanced_accuracy,test_f1,test_precision,test_recall):
    '''
    Function that exports nn results to csv table
    input: - table_name - name of the csv file
           - init_features
           - batch_size
           - test statistics: balanced accuracy, recall, precision, and f1
           
    '''
    result = [init_features,batch_size,test_balanced_accuracy,test_f1,test_precision,test_recall]
    
    # We create a csv where we save our nn results
    with open(table_name,'a',encoding='UTF8',newline='') as f:

      writer = csv.writer(f)
      
      # append our result
      writer.writerow(result)

def randomized_search_nn(train_data,test_data,labels,classes,init_features,batch_size,init_lr,iterations=5, epochs=30,verbose=True,gpu=True,table_name='UNet_results.csv'):

      '''
      This function performs randomized search on the hyperparameters: init_features and batch_size for our neural
      network. We also create a csv file where we export our testing results.
      input: train_data, test_data, labels = [0,1,2,..], classes = ['Agriculture','Forestland',..]
             init_features,batch_size - we need to pass a list for the possible choices
             iterations - how many different combinations between then above sets we want.
             epoches, verbose - if we want to print training loss after each epoch
             table_name - the name of the csv file
             init_lr - initial learning rate
      '''
      
      # We create a csv where we save our nn results
      with open(table_name,'w',encoding='UTF8',newline='') as f:

          writer = csv.writer(f)

          writer.writerow(['init_features','batch_size','balanced_accuracy','f1','recall','precision'])
      
      # counter to keep track on the iterations in our loop
      counter = 0 
      
      #list of all the possible permutations between init_features and batch_size
      parameters = list(itertools.product(init_features,batch_size,init_lr))
      
      # we loop until we have elements in the parameters list or until we have less iterations
      # than the given function variable 'iterations'

      while len(parameters)>0 or (counter < iterations):

            # pick random (param1,param2) from parameters
            choice = random.choice(parameters)

            choice_init_features = choice[0]
            choice_batch_size = choice[1]
            choice_init_lr = choice[2]
            
            # we start training our model
            model,test_true,test_predictions = train_validate_model(train_data,test_data,labels,classes,init_features=choice_init_features,
                                                    batch_size=choice_batch_size,epochs=epochs,init_lr = choice_init_lr,
                                                    verbose=verbose,gpu=gpu)
            
            # Empty line for the verbose of the next model
            print(' ')

            # we calculate the test statistics after we looped through all epochs
            test_balanced_accuracy,test_f1,test_precision,test_recall = test_statistics(model,test_true,test_predictions,verbose=False)

            # we save the results to excell table
            export_to_csv(table_name = table_name,init_features = choice_init_features,batch_size = choice_batch_size, 
                          test_balanced_accuracy = test_balanced_accuracy, test_f1 = test_f1,
                          test_precision = test_precision,test_recall = test_recall)
            
            # remove this element from our parameters so we don't test the same parameter again
            parameters.remove(choice)


def create_map(model,raster_image,raster_subimage,raster_subimage_classifier,
             windows_height,windows_width,origin_x,origin_y,bands,gpu=True):
    
    '''
    This function takes as an input a raster image and a pre-trained deep learning model 
    (UNet) and outputs a classification map.
    input: model - our best model
           raster_image - path to the raster image
           raster_subimage - path to the subimage of the raster that would be created
           raster_subimage_classifier - path to the map related to the raster_subimage
           also the windows origin coordinates and whether to use gpu
           bands - number of bands for our image

    
    output: raster_subimage and raster_subimage_classifier <-- the map, to the chosen 
            locations

    '''
    
    # Load the raster image
    src = rasterio.open(raster_image, 'r')
    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        count=1,
      )
    
    # copy its meta file
    meta = src.meta.copy()
    
    # create a windows with origin 
    window = rasterio.windows.Window(origin_x,origin_y,windows_height,windows_width)
    transform = rasterio.windows.transform(window, src.transform)

    # read the data from this window
    data = src.read(window=window)

    if gpu == True:
        # convert to tensor
        data_tensor = torch.FloatTensor(data).cuda()
    else:
        data_tensor = torch.FloatTensor(data)

    # reshape the data ( we treat it as a batch with 1 sample, so we add 1 dimension)
    data_tensor_reshaped = data_tensor.view(1,bands,windows_height,windows_width)

    # run the model and get the outputs
    y_pred = model.forward(data_tensor_reshaped)

    # find the predictions
    map = torch.argmax(y_pred, dim=1)

    # put the map on cpu and convert it to numpy
    map_np = map.cpu().detach().numpy()

    # setting the appropriate meta 
    meta['count'] = 1
    meta['height']= windows_height
    meta['width'] = windows_width
    meta['transform'] = transform

    # save the map
    with rasterio.open(raster_subimage_classifier, 'w', **meta) as outds:
                outds.write(map_np)

    # change the meta for the raster
    meta['count'] = bands

    # save the the subraster image
    with rasterio.open(raster_subimage, 'w', **meta) as outds:
                outds.write(src.read(window=window))

def cut_img(im,perc):
  '''
  This function cuts each polygon of an image with perc.
  The cutting is with respect to the y-axis
  '''
  
  im = im.copy()
  
  # we will use this to edit the polygons mask
  im_copy = im.copy()
  
  # this is the image we return - the cut image
  new_img = np.full((256,256),-999)

  # we swap -999,-1->0,  0->500, before using ndimage.label
  mask_999 = (im==-999)
  mask_0 = (im==0)
  mask_1 = (im==-1)

  im_copy[mask_1]=0
  im_copy[mask_999]=0
  im_copy[mask_0]=500

  s = ndimage.generate_binary_structure(2,2)
  
  # the labeled polygons from the image, and the number of diff polygons
  im_pol,n_pols = ndimage.label(im_copy.reshape((256,256)),structure =s)

  # loop through the id of different polygons 1-n_pols
  for pol_id in np.arange(1,n_pols+1):

      # find the indexes in the labeled polygons where the label is equal to pol_id
      # Thus, these are indexes for only 1 polygon
      inds = np.where(im_pol==pol_id)
      
      # number of indexes in the above mentioned polygon
      n = len(inds[0])
      
      # we remove the last perc indexes/ cut the polygon indexes / cut the polygon
      cut = int(np.ceil((1-perc)*n))
      inds_new = (inds[0][:cut],inds[1][:cut])

      im = im.reshape((256,256))

      # we find the image value that corresponds to the current polygon
      val = im[inds_new][0]

      # we put in our return image the indexes of the current cut polygon, the
      # mentioned value on the previous line
      new_img[inds_new] = val


  return new_img.reshape((1,256,256))

def cut_labels(y,perc):
  '''
  This function uses the funtion cut_img, to cut each image from
  a label dataset, and return the new cut label dataset
  '''
  y = y.copy()

  for i,img in enumerate(y):

      # assign to the current index the cut image of the current image
      y[i] = cut_img(img,perc)

  
  return y 