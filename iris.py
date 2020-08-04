# k-nearest-neighbors-algorithm-exercise

iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
print("Attributes:")
print(X)
print("\nLabels:")
print(y)


from sklearn.model_selection import train_test_split
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("\n70% train data:")
print(X_train)
print(y_train)
print("\n30% test data:")
print(X_test)
print(y_test)


from sklearn.model_selection import train_test_split
iris = pd.read_csv("iris.csv")
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Species = le.fit_transform(iris.Species)
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("\n80% train data:")
print(X_train)
print(y_train)
print("\n20% test data:")
print(X_test)
print(y_test)


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
'''
print("\n70% train data:")
print(X_train)
print(y_train)
print("\n30% test data:")
print(X_test)
print(y_test)
'''
#Create KNN Classifier
#Number of neighbors to use by default for kneighbors queries.
knn = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
print("Response for test dataset:")
y_pred = knn.predict(X_test)
print(y_pred)


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
# Calculate the accuracy of the model 
print("Accuracy of the model:")
print(knn.score(X_test, y_test))


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
# Calculate the accuracy of the model for different values of k
for i in np.arange(1, 10):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    print("For k = %d accuracy is"%i,knn2.score(X_test,y_test))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  
from sklearn import metrics

iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
a_index=list(range(1,11))
a=pd.Series()
# Calculate the accuracy of the model for different values of k
for i in np.arange(1, 10):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    print("For k = %d accuracy is"%i,knn2.score(X_test,y_test))
# Visual presentation: Various values of n for K-Nearest nerighbours
print("\nVisual presentation: Various values of n for K-Nearest nerighbours:")    
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
plt.plot(a_index, a)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  

iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)

print("Preliminary model score:")
print(knn.score(X_test,y_test))

no_neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Visualization of k values vs accuracy

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()



# iris-basic-exercise

data = pd.read_csv("iris.csv")
print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))


iris_data = pd.read_csv("iris.csv")
print("\nKeys of Iris dataset:")
print(iris_data.keys())
print("\nNumber of rows and columns of Iris dataset:")
print(iris_data.shape) 


iris = pd.read_csv("iris.csv")
print(iris.info())


from scipy import sparse
eye = np.eye(4)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)


data = pd.read_csv("iris.csv")
print(data.describe())


data = pd.read_csv("iris.csv")
print("Observations of each species:")
print(data['Species'].value_counts()) 


data = pd.read_csv("iris.csv")
print("Original Data:")
print(data.head())
new_data = data.drop('Id',axis=1)
print("After removing id column:")
print(new_data.head()) 


data = pd.read_csv("iris.csv")
print("Original Data:")
print(data.head())
new_data = data.drop('Id',axis=1)
print("After removing id column:")
print(new_data.head())
x = data.iloc[:, [1, 2, 3, 4]].values
print(x) 



# iris-visualization-exercise

import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
iris.describe().plot(kind = "area",fontsize=16, figsize = (15,8), table = True, colormap="Accent")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Iris Dataset")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Species',data=iris)
plt.title("Iris Species Count")
plt.show()


import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
iris['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.title("Iris Species %")
plt.show()


import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
# Drop id column
new_data = iris.drop('Id',axis=1)
new_data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, color='blue') 
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', kind="hex", color="red", data=iris)
plt.show()


import pandas as pd
import seaborn as sns
iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', kind="kde", color='cyan', data=iris)  
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', kind="reg", color='red', data=iris) 
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
sns.jointplot("SepalLengthCm", "SepalWidthCm", data=iris, color="b").plot_joint(sns.kdeplot, zorder=0, n_levels=6) 
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
g = sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$SepalLength(Cm)$", "$SepalWidth(Cm)$") 
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
g = sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$SepalLength(Cm)$", "$SepalWidth(Cm)$") 
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
sub=iris[iris['Species']=='Iris-setosa']
sns.kdeplot(data=sub[['SepalLengthCm','SepalWidthCm']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Sepal Length cm')
plt.ylabel('Sepal Width cm')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
sns.kdeplot(data=sub[['PetalLengthCm','PetalWidthCm']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Petal Length Cm')
plt.ylabel('Petal Width Cm')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
sns.kdeplot(data=sub[['PetalLengthCm','PetalWidthCm']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Petal Length Cm')
plt.ylabel('Petal Width Cm')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, 0:4]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() 


import seaborn as sns 
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
box_data = iris #variable representing the data array
box_target = iris.Species #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(2,15)})


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import preprocessing

# import iris.csv
iris = pd.read_csv("iris.csv")
# Converting string labels into numbers.
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Species = le.fit_transform(iris.Species)
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

fig = plt.figure(1, figsize=(7, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()



# logistic-regression-exercise

data = pd.read_csv("iris.csv")
print('Iris-setosa')
setosa = data['Species'] == 'Iris-setosa'
print(data[setosa].describe())
print('\nIris-versicolor')
setosa = data['Species'] == 'Iris-versicolor'
print(data[setosa].describe())
print('\nIris-virginica')
setosa = data['Species'] == 'Iris-virginica'
print(data[setosa].describe())


import matplotlib.pyplot as plt
from sklearn import preprocessing
iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
#Convert Species columns in a numerical column of the iris dataframe
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Species = le.fit_transform(iris.Species)
x = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
plt.scatter(x[:,0], x[:, 3], c=y, cmap ='flag')
plt.xlabel('Sepal Length cm')
plt.ylabel('Petal Width cm')
plt.show()


from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

iris = pd.read_csv("iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y_test))



