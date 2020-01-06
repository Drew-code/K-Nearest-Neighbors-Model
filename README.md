# K-Nearest-Neighbors-Model
Quick Start Guide to K-NearestNeighbors Model in Python  
  
## Introduction  
K-nearest neightbors is a gouping algorithum that picks a point in the data set, then looks for pattens in the data points  
closest to that one. It then uses those patterns to determine what makes those points similar. Creating this model is very  
similar to the other models in scikit-learn with a couple of extra steps. We will need to scale the features and then test  
for the best number of neighbors used.
 
## Prerequisites
1. Python 3.7 or newer  
2. Scikit-Learn module for Python  
3. Pandas module for Python  
4. Numpy module for Python  
5. Matplotlib module for Python
  
## Walkthough  
Start by importing all modules you will need at the beginning of your script. These include: Pandas, Scikit-Learn,  
Matplotlib,  and seaborn.  
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
```  
Next import the csv file and clean it up a little. Below we are importing the csv file as an object "df".  
Once again we print the head of the data along with the info to get an idea of what the data looks like.  

```
df = pd.read_csv(r"D:\Datasets\Amazon_Stock.csv",index_col=0)
df.head()
df.info()
```  
With this model, we need to add in a step to set all of the variables to the same scale with a scaler. We do this  
with all variables except the "Y" or dependant variable. In this example that is 'Change'. Then a new dataframe is created  
with all of the transformed features. As with other models, it is now time to set the X and y variables.
```
scaler = StandardScaler()
scaler.fit(df.drop('Change',axis=1)) #set variables except target to a scale
scaled_features = scaler.transform(df.drop('Change',axis=1)) #transformed those feature or variables to scaled features variables
df_feat = pd.DataFrame(scaled_features) # created a new data frame with the scaled features and not the target
X = df_feat #set our X
y = df['Change'] # set our Y
```  
Now that the variables have been established, it is time to split the data into two parts. The training set  
is what our model is going to look at and learn from. The model will then try to apply what it has learned to the test data.  
The "test_size" argument, is set to 0.3 in this example. That means that 70% of the data will be used to train the model  
and the remaining 30% will be used to test how accurate the model is. The "random_state" parameter sets the random  
number generator. This makes it so we are able to replicate results.  
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


With all of the data cleaned and prepped, all that is left to do is create the model and start training it.  
First, create an instance of the model that is being used, in this example it is Kneighborsclassifier. We than have to enter the number of neighbors we want the model to use when making a prediction. This is just a starting point, a better number will be   
used after the first run. Next we fit or train the model, followed by prediction off of the test data we split off earlier.
```
knn = KNeighborsClassifier(n_neighbors=1) #chose the number of points we wanted included to make a prediction
knn.fit(X_train,y_train) #trained with that number of ploints
pred = knn.predict(X_test) #predicted based on known results

```  
To see how accurate the model is, we need to print out a couple of reports. The main report I use is the confusion matrix.  
This shows the number of true positives, false positives, true negatives and false negatives. It also gives an overall accuracy  
score. There are other great reports you can use in conjuction with this, such as the classification report. This is great  
if you have more than two groups the result can end up in.  
```
print("Confusion Matrix")
print(confusion_matrix(y_test,pred)) #looked at results of the first model
print("Classification Report")
print(classification_report(y_test,pred))
```  
After getting a baseline of accuracy using n_neighbors set to 1. We then can create a loop to test error_rate at each neighbor  
from 1 to 40. That information can then be ploted on a graph to help us visualize the results. The point where the error rate  
is the lowest and levels out, is the number of neighbors that should provide the best accuracy in the model.  
```
error_rate = []

for i in range(1,40): ##created a for loop to test more k values to find most accurate one
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',ls='dashed',marker='o', #plotted the k values to find our best ones
         markerfacecolor='red',markersize=10)
plt.title('Error rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```  
After the initial model is evaluated, a new model must in instantiated using the newly accquired number of neighbors.  
the model then has to be re-trained again and then predictions called on the train data that was split earlier.  
Then it is time to compare results using the same metrics as before. This model takes some tweaking to get accurate.  
As you will see in the graph that was created, this dataset is not a great example for this model. Feel free to use your own  
data and see what you can coem up with.  
```
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
```








