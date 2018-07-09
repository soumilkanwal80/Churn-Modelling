import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as pl
#Read Data
dataset=pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Preprocessing Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
l1 = LabelEncoder()
X[:, 1]=l1.fit_transform(X[:, 1])
l2 = LabelEncoder()
X[:, 2]=l2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]

#Spliiting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Building the Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

 =============================================================================
 clf=Sequential()
 clf.add(Dense(units=12, kernel_initializer = 'uniform', activation='relu'))
 clf.add(Dropout(rate = 0.15))
 clf.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
 clf.add(Dropout(rate = 0.15))
 clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
 
 clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
 clf.fit(X_train, Y_train, batch_size=10, epochs=100)
 
 Y_pred=clf.predict(X_test)
 Y_pred=(Y_pred>0.5)
 
 from sklearn.metrics import confusion_matrix
 cm=confusion_matrix(Y_test, Y_pred)
 
 =============================================================================

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score, StratifiedKFold
#
#def build_clf():
#    clf=Sequential()
#    clf.add(Dense(units=12, kernel_initializer = 'uniform', activation='relu'))
#    clf.add(Dropout(rate = 0.15))
#    clf.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
#    clf.add(Dropout(rate = 0.15))
#    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#    return clf  
#    
#clf=KerasClassifier(build_fn=build_clf, batch_size=10, epochs=100)
#kfold = StratifiedKFold(n_splits = 2, shuffle=True, random_state=7)
#accuracies=cross_val_score(estimator=clf, X=X_train, y=Y_train, cv=kfold, n_jobs=-1)
#mean=accuracies.mean()
#dev= accuracies.std()
#print(mean)
#print(dev)

from keras.models import model_from_json, load_model

clf.save('my_model.h5')

model_json = clf.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

clf.save_weights("./model.h5")
print("saved model..! ready to go.")

        