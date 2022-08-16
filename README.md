# Creating classification maps from Scotland dataset

Our goal is to create classification maps from dataset which 
consists of a raster image of a region in Scotland and a vector (geojson)
file which consists of polygons related to this raster image. We are
working with sparse data. Thus, not every pixel in our raster image will 
have a label assigned to it. 

Technologies we use:
Python - PyTorch, Scikit Learn, NumPy, Pandas, GeoPandas, Seaborn

QGIS - mostly for visualization and editing/creating polygons.
                  
We use both machine learning and deep learning models and compare their results using
5 test statistics - balanced_accuracy, recall, precision, f1, standart accuracy.

ML models we use: * LightGBM
                  * XGBoost
                  * Random Forest
DL models we use we use:  * UNet architecture

Results: Our best deep learning model outperforms our best machine learning
model (LightGBM) as it has higher test results. In addition, when we saw how the
generated classification maps for the best ML and best DL method look, we saw that
the DL map captures linear features like rivers and roads a lot better. However, overall
we can conclude that both models produce good maps.
