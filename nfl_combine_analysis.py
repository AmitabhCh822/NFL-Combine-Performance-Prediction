import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

# pd.read_fwf() reads the .dat file by dividing the data into columns
df1 = pd.read_fwf('nfl_combine_2014.dat.txt', index_col=0)

#Converting the dat file into csv file
df1.to_csv('nfl.csv')
df = pd.read_csv('nfl.csv')

#Deleting the first three columns which are not be used in the analysis
df.drop(df.iloc[:,[0,1,2]], axis=1, inplace=True)

#Labeling the columns (Code given by Dr. Treu)
df.columns = ['Grade', 'Height', 'Length', 'Weight', '40Yard', 'BenchPress',
              'VerticalJump', 'BroadJump', '3Cone', '20Yard', 'Extra']

#Eliminating rows with missing values (Code retrieved from Stack Overflow)
df = df.dropna()  

#Converting the Grade attribute to nominal from numeric
df['Grade'] = np.where(df['Grade'] > df['Grade'].median(),'good','bad') #Code retrieved from Stack Overflo)

X = df.iloc[:,[1,2,3,4,5,6,7,8,9]]  #All the attributes other than 'Grade' are the predictors
y = df.iloc[:,0]  #'Grade' is the output attribute

#Normalizing the predictors' data
X_normalized = X.apply(lambda x: (x -min(x))/(max(x)-min(x)))

#Creating the train and test data
X_train, X_test, y_train, y_test = train_test_split(X_normalized,y,test_size=0.30,random_state=48)

#Creating KNN Classfier model
knn = KNeighborsClassifier(n_neighbors=3)

#Fitting the training data
knn.fit(X_train,y_train)

#Predicting on the test data
pred = knn.predict(X_test)

print("The confusion matrix of the knn model: \n", confusion_matrix(y_test,pred), "\n")
print("The classifiction report of the knn model: \n", classification_report(y_test,pred), "\n")
print("The accuracy of the knn model with k = 3: \n", accuracy_score(y_test,pred), "\n")