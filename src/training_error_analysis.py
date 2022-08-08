import numpy as np
from geojson import dump 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier 
from prepare_data_for_training_ML_ANN import further_train_test_split
#import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

def bad_pixels_per_test_polygon(test_polygons_ids,dataframe,preds,y_test,class_dict,top_percentage=0.5):
    '''
    Input: id's of test polygons, dataframe, predictions,y_test,class_dict - dictionary with our classes,
           top_percentage - threshold for the bad polygons.
    Output: - Dictionary with sorted by number of missclassified pixels ids of test polygons as percentage.
            - Dataframe with these polygons. In this dataframe we include the percentage and also for each
              polygon we find to which is the class other than its own class, that we mostly classify it to.
            - A list of the polygon's indexes
            
            Also we create a geojson file that with the polygons 
    '''
    

    # INITIALIZE A DICTIONARY WITH KEYS <-> THE UNIQUE TEST_POLYGON_IDS, AND VALUES = 0 
    # In this dictionary we will store the number of bad pixels we have in each test polygon
    unique_ids_test_polygon = np.unique(test_polygons_ids)
    test_polygons_bad_pixels = dict.fromkeys(unique_ids_test_polygon,0)


    # Count the pixels per test polygon
    pixels_per_test_polygon = dict.fromkeys(unique_ids_test_polygon,0)

    for item in test_polygons_ids:

        pixels_per_test_polygon[item]+=1

    # Loop through y_test, the predicted labels - preds, and the test_polygons_ids
    # if the value of y_test and preds is not the same then we have a bad pixel
    # so add 1 to the polygons_errors
    for pred,true,index in zip(preds,y_test,test_polygons_ids):
        if pred!=true:
         test_polygons_bad_pixels[index]+=1


    # We find the percentage of bad pixels over all pixels in each test polygon
    percentage_bad_pixels_per_test_polygon = {index: round(test_polygons_bad_pixels[index]/pixels_per_test_polygon[index],3) for index in unique_ids_test_polygon}
    
    # WE SORT 
    percentage_bad_pixels_per_test_polygon = {k: v for k, v in sorted(percentage_bad_pixels_per_test_polygon.items(), key=lambda item: item[1],reverse=True)}

    # list of the ids that have more than 50% error.
    list50 = [k for k, v in percentage_bad_pixels_per_test_polygon.items() if v > top_percentage]

    # Let's peek into the worst polygons
    worst_test_polygons = dataframe.iloc[list50].copy()
    
    # We put the percentage of badness in their description (useful for QGIS)
    worst_test_polygons['confused percentage'] = [percentage_bad_pixels_per_test_polygon[index] for index in worst_test_polygons.id]

    # Create a reversed class dictionary 0->Agricultur, 1-> Built, ...
    reversed_class_dict = {v:k for k,v in class_dict.items()}
    reversed_class_dict

    # Create a column that holds this most confused class that we have for our polygon.
    worst_test_polygons['most_confused_class'] = np.zeros(len(worst_test_polygons.id))

    #Loop through the indexes of the worst test polygons
    for worst_index in worst_test_polygons.id:
            
            # for each bad index ( index from our dataset 'worst_test_polygons, and so for each bad test polygon), we see the frequency of the missclassifications.
            confusion_freq = dict.fromkeys([0,1,2,3,4,5],0)
            
            for (bad_id,true,pred) in zip(test_polygons_ids,y_test,preds):
            
                    if bad_id == worst_index and true!=pred:
                    
                            confusion_freq[pred]+=1
            
            # WE SORT our confusion_frequency
            confusion_freq = {k: v for k, v in sorted(confusion_freq.items(), key=lambda item: item[1],reverse=True)}
            confusion_freq

            # Take the first item of the dictionary. This is our most confused class for this polygon.
            most_confused_class_numerical = next(iter(confusion_freq))

            # Now we convert the item of this dictionary to character form.
            most_confused_class_character = reversed_class_dict[most_confused_class_numerical]

            # We add it to a new column in our dataframe worst_test_polygons.

            worst_test_polygons.at[worst_index,'most_confused_class'] = most_confused_class_character


        # Create a geojson file from the bad_gpd, so we can visualize it in QGIS
    with open('worst_test_polygons.geojson', 'w') as f:
            dump(worst_test_polygons, f)

    
    return (worst_test_polygons,percentage_bad_pixels_per_test_polygon,list50)


    
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

def cross_validate_ml(df,model,raster_file,class_dict,labels='numeric_class',seed=False,weight=False,folds = 4):

  '''
  This function performs a stratified cross validation for a machine learning method from sklearn
  input: 
  output: - balanced accuracies based on the number of folds
          - accuracies
          - recalls
          - precisions
          - f1s
          - confusion matrices
  '''
  results_bal_acc = []
  results_prec=[]
  results_recall=[]
  results_f1=[]
  results_accuracy = []
  cm_s = []
  
  # Create the splits
    
  # Our split
  if seed!=False:  
      np.random.seed(seed)

  skf = StratifiedKFold(n_splits=folds,shuffle=True)
  skf.get_n_splits(df, df[labels])
  
  for train_index, test_index in skf.split(df,df[labels]):
        
         # we get our training and testing set of vectors
         training_set_vectors =  df.iloc[train_index]
         testing_set_vectors = df.iloc[test_index]
         
         # we will need that for our confusion matrix
         classes = np.unique(training_set_vectors.classes)
         
         # we rasterize them and make a further split to obtain X_train, y_train, X_test, y_test
         X_train,y_train,X_test,y_test = further_train_test_split(training_set_vectors,testing_set_vectors,raster_file,class_dict)
         if len(np.unique(y_test))!=len(np.unique(y_train)):
             print('Error - y_test does not contain all classes')
             return 'Error - y_test does not contain all classes'

         # calculate class weights to allow for training on inbalanced training samples
         labels, counts = np.unique(y_train, return_counts=True)
         if weight == True:
            class_weight_dict = dict(zip(labels, 1 / counts))
            model.set_params(class_weight = class_weight_dict)
                         
         # train the model
         model.fit(X_train,y_train)

         # evaluate the model
         preds_model = model.predict(X_test) # generate predictions
  
         # calculate the test statistics
         bal_acc,recall,precision,f1,acc = test_statistics(y_test,preds_model,verbose=False)

         # append them to our lists
         results_bal_acc.append(bal_acc)
         results_prec.append(precision)
         results_recall.append(recall)
         results_f1.append(f1)
         results_accuracy.append(acc)  

         cm_model = confusion_matrix(y_test, preds_model, labels=labels)
         plot_cm(cm_model,classes)
         plt.show()
         cm_s.append(cm_model)
  
  # the final confusion matrix
  sum_cm = np.zeros(cm_s[0].shape)
  for matrix in cm_s:
    sum_cm +=matrix
  mean_cm = sum_cm/folds  

  return np.array(results_accuracy).mean(),np.array(results_bal_acc).mean(),np.array(results_prec).mean(),np.array(results_recall).mean(),np.array(results_f1).mean(),mean_cm

def lgb_cross_validate_ml(df,raster_file,class_dict,labels='numeric_class',seed=False,folds = 4):

  '''
  This function performs a stratified cross validation for a machine learning method from sklearn
  input: 
  output: - balanced accuracies based on the number of folds
          - recalls
          - precisions
          - f1s
          - confusion matrices
  '''
  results_bal_acc = []
  results_prec=[]
  results_recall=[]
  results_f1=[]
  results_accuracy = []
  cm_s = []
  
  # Create the splits
  if seed!=False:  
      np.random.seed(seed)

  skf = StratifiedKFold(n_splits=folds,shuffle=True)
  skf.get_n_splits(df, df[labels])
  
  for train_index, test_index in skf.split(df,df[labels]):
        
         # we get our training and testing set of vectors
         training_set_vectors =  df.iloc[train_index]
         testing_set_vectors = df.iloc[test_index]

           # we will need that for our confusion matrix
         classes = np.unique(training_set_vectors.classes)
         
         # we rasterize them and make a further split to obtain X_train, y_train, X_test, y_test
         X_train,y_train,X_test,y_test = further_train_test_split(training_set_vectors,testing_set_vectors,raster_file,class_dict)
         if len(np.unique(y_test))!=len(np.unique(y_train)):
             print('Error - y_test does not contain all classes')
             return 'Error - y_test does not contain all classes'
        
         # calculate class weights to allow for training on inbalanced training samples
         labels, counts = np.unique(y_train, return_counts=True)
         class_weight_dict = dict(zip(labels, 1 / counts))
          
         
         # DEFINE THE LGBM
         model = lgb.LGBMClassifier(
                objective='multiclass',
                class_weight = class_weight_dict,
                num_class = len(class_weight_dict)
                )
    
         # train the model
         model.fit(X_train,y_train)

         # evaluate the model
         preds_model = model.predict(X_test) # generate predictions
  
         # calculate the test statistics
         bal_acc,recall,precision,f1,acc = test_statistics(y_test,preds_model,verbose=False)

         # append them to our lists
         results_bal_acc.append(bal_acc)
         results_prec.append(precision)
         results_recall.append(recall)
         results_f1.append(f1) 
         results_accuracy.append(acc)   

         cm_model = confusion_matrix(y_test, preds_model, labels=labels)
         plot_cm(cm_model,classes)
         plt.show()
         cm_s.append(cm_model)
  
  # the final confusion matrix
  sum_cm = np.zeros(cm_s[0].shape)
  for matrix in cm_s:
    sum_cm +=matrix
  mean_cm = sum_cm/folds  

  return np.array(results_accuracy).mean(),np.array(results_bal_acc).mean(),np.array(results_prec).mean(),np.array(results_recall).mean(),np.array(results_f1).mean(),mean_cm

def rf_cross_validate_ml(df,raster_file,class_dict,labels='numeric_class',seed=False,folds = 4):

  '''
  This function performs a stratified cross validation for a machine learning method from sklearn
  input: 
  output: - balanced accuracies based on the number of folds
          - accuracies
          - recalls
          - precisions
          - f1s
          - confusion matrices
  '''
  results_bal_acc = []
  results_prec=[]
  results_recall=[]
  results_f1=[]
  results_accuracy = []
  cm_s = []
  
  # Create the splits
  if seed!=False:  
      np.random.seed(seed)

  skf = StratifiedKFold(n_splits=folds,shuffle=True)
  skf.get_n_splits(df, df[labels])
  
  for train_index, test_index in skf.split(df,df[labels]):
        
         # we get our training and testing set of vectors
         training_set_vectors =  df.iloc[train_index]
         testing_set_vectors = df.iloc[test_index]

           # we will need that for our confusion matrix
         classes = np.unique(training_set_vectors.classes)
         
         # we rasterize them and make a further split to obtain X_train, y_train, X_test, y_test
         X_train,y_train,X_test,y_test = further_train_test_split(training_set_vectors,testing_set_vectors,raster_file,class_dict)
         if len(np.unique(y_test))!=len(np.unique(y_train)):
             print('Error - y_test does not contain all classes')
             return 'Error - y_test does not contain all classes'
   
         # calculate class weights to allow for training on inbalanced training samples
         labels, counts = np.unique(y_train, return_counts=True)
         class_weight_dict = dict(zip(labels, 1 / counts))
          
         
         # DEFINE THE MODEL
         model = RandomForestClassifier(n_estimators = 400, max_depth = 20,class_weight=class_weight_dict)
    
         # train the model
         model.fit(X_train,y_train)

         # evaluate the model
         preds_model = model.predict(X_test) # generate predictions
  
         # calculate the test statistics
         bal_acc,recall,precision,f1,acc = test_statistics(y_test,preds_model,verbose=False)

         # append them to our lists
         results_bal_acc.append(bal_acc)
         results_prec.append(precision)
         results_recall.append(recall)
         results_f1.append(f1)
         results_accuracy.append(acc)  

         cm_model = confusion_matrix(y_test, preds_model, labels=labels)
         plot_cm(cm_model,classes)
         plt.show()
         cm_s.append(cm_model)
  
  # the final confusion matrix
  sum_cm = np.zeros(cm_s[0].shape)
  for matrix in cm_s:
    sum_cm +=matrix
  mean_cm = sum_cm/folds  

  return np.array(results_accuracy).mean(),np.array(results_bal_acc).mean(),np.array(results_prec).mean(),np.array(results_recall).mean(),np.array(results_f1).mean(),mean_cm

def xgb_cross_validate_ml(df,raster_file,class_dict,labels='numeric_class',seed=False,folds = 4):

  '''
  This function performs a stratified cross validation for a machine learning method from sklearn
  input: 
  output: - balanced accuracies based on the number of folds
          - accuracies
          - recalls
          - precisions
          - f1s
          - confusion matrices
  '''
  results_bal_acc = []
  results_prec=[]
  results_recall=[]
  results_f1=[]
  results_accuracy = []
  cm_s = []
  
  # Create the splits
  if seed!=False:  
      np.random.seed(seed)

  skf = StratifiedKFold(n_splits=folds,shuffle=True)
  skf.get_n_splits(df, df[labels])
  
  for train_index, test_index in skf.split(df,df[labels]):
        
         # we get our training and testing set of vectors
         training_set_vectors =  df.iloc[train_index]
         testing_set_vectors = df.iloc[test_index]

           # we will need that for our confusion matrix
         classes = np.unique(training_set_vectors.classes)
         
         # we rasterize them and make a further split to obtain X_train, y_train, X_test, y_test
         X_train,y_train,X_test,y_test = further_train_test_split(training_set_vectors,testing_set_vectors,raster_file,class_dict)
         if len(np.unique(y_test))!=len(np.unique(y_train)):
             print('Error - y_test does not contain all classes')
             return 'Error - y_test does not contain all classes'
         
         # calculate class weights to allow for training on inbalanced training samples
         labels, counts = np.unique(y_train, return_counts=True)
         class_weight_dict = dict(zip(labels, 1 / counts))
          
         
         # DEFINE THE MODEL
         model =  XGBClassifier(
              objective='multiclass',
              class_weight = class_weight_dict,
              num_class = len(class_weight_dict),
              metric = 'multi_logloss')
    
         # train the model
         model.fit(X_train,y_train)

         # evaluate the model
         preds_model = model.predict(X_test) # generate predictions
  
         # calculate the test statistics
         bal_acc,recall,precision,f1,acc = test_statistics(y_test,preds_model,verbose=False)

         # append them to our lists
         results_bal_acc.append(bal_acc)
         results_prec.append(precision)
         results_recall.append(recall)
         results_f1.append(f1)
         results_accuracy.append(acc)  

         cm_model = confusion_matrix(y_test, preds_model, labels=labels)
         plot_cm(cm_model,classes)
         plt.show()
         cm_s.append(cm_model)
  
  # the final confusion matrix
  sum_cm = np.zeros(cm_s[0].shape)
  for matrix in cm_s:
    sum_cm +=matrix
  mean_cm = sum_cm/folds  

  return np.array(results_accuracy).mean(),np.array(results_bal_acc).mean(),np.array(results_prec).mean(),np.array(results_recall).mean(),np.array(results_f1).mean(),mean_cm