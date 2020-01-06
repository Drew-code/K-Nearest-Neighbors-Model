import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv(r"D:\Datasets\Amazon_Stock.csv",index_col=0)
df.head()
df.info()

scaler = StandardScaler()
scaler.fit(df.drop('Change',axis=1)) #set variables except target to a scale
scaled_features = scaler.transform(df.drop('Change',axis=1)) #transformed those feature or variables to scaled features variables
df_feat = pd.DataFrame(scaled_features) # created a new data frame with the scaled features and not the target
X = df_feat #set our X
y = df['Change'] # set our Y

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101) #decided the split the of the data


knn = KNeighborsClassifier(n_neighbors=1) #chose the number of points we wanted included to make a prediction
knn.fit(X_train,y_train) #trained with that number of ploints
pred = knn.predict(X_test) #predicted based on known results


print("Confusion Matrix")
print(confusion_matrix(y_test,pred)) #looked at results of the first model
print("Classification Report")
print(classification_report(y_test,pred))

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

knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))