# Creating classification maps from Scotland dataset

Our goal is to create classification maps from dataset which 
consists of a raster image of a region in Scotland and a vector (geojson)
file which consists of polygons related to this raster image. We are
working with sparse data. Thus, not every pixel in our raster image will 
have a label assigned to it. 

Technologies we use:

Python - PyTorch, Scikit Learn, NumPy, Pandas, GeoPandas, Seaborn.
QGIS - mostly for visualization and editing/creating polygons.
                  
We use both machine learning and deep learning models and compare their results using
5 test statistics - balanced_accuracy, recall, precision, f1, standart accuracy.

ML models we use: LightGBM , XGBoost, Random Forest.

DL models we use we use:  UNet architecture.

Results: Our best deep learning model outperforms our best machine learning
model (LightGBM) as it has higher test results. In addition, when we saw how the
generated classification maps for the best ML and best DL method look, we saw that
the DL map captures linear features like rivers and roads a lot better. However, overall
we can conclude that both models produce good maps.

Scotland Raster Area:

![image](https://user-images.githubusercontent.com/77898273/184901038-00921e14-1528-4956-abf5-ebb6e7b6233a.png)

Scotland Raster Image:

![image](https://user-images.githubusercontent.com/77898273/184900618-37720bb2-8316-4f5b-9296-c4d55b42702d.png)

Scotland Polygons:

![image](https://user-images.githubusercontent.com/77898273/184901404-0cfe8a7d-6093-482f-b7bc-ef7eb8263aa6.png)

Best DL classification map of the region above:

<img src="https://user-images.githubusercontent.com/77898273/184902824-327e0362-2cb4-401a-b97a-e8b7a0a26793.png" height="800" width="800">

Best ML classification map of the region above:

<img src="https://user-images.githubusercontent.com/77898273/184902915-ff2ea91f-2bba-43f1-82c4-882858647002.png" height="800" width="800">

Stats Best DL:

<img src="https://user-images.githubusercontent.com/77898273/184906056-2a618618-107a-4c58-9f24-3d3c343e4a6c.png" width="800">

Stats Best ML:

<img src="https://user-images.githubusercontent.com/77898273/184903840-6dde4f22-7edb-472e-b91c-65cbbb670561.png" width="800">

Here by time cost we mean the training time for our model for 1 fold. We have 4 folds in total.
Thus this is the time cost for a model with 75%-25% train-test split.
UNet architecture:
![image](https://user-images.githubusercontent.com/77898273/184901603-8daeae8d-89ca-460b-ae3e-b372d3e87376.png)




