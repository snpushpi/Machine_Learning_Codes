import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
my_data = pd.read_csv("drug200.csv") 
#X is the feature matrix, y be the target value
X =  my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values  
y = my_data["Drug"]
# now we will do some labeling, essentially we will do categorization 
from sklearn import preprocessing 
le_sex = preprocessing.LabelEncoder() 
le_sex.fit(['F','M']) 
X[:,1]=le_sex.fit_transform(X[:,1]) # we are transforming data , from alphanumeric labels to numeric ones

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH']) 
X[:,2]=le_BP.fit_transform(X[:,2])

le_Chol = preprocessing.LabelEncoder() 
le_Chol.fit(['NORMAL','HIGH']) 
X[:,3]=le_Chol.fit_transform(X[:,3]) 

from sklearn.model_selection import train_test_split 
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y, test_size = 0.3, random_state = 3) 
drugTree = DecisionTreeClassifier(criterion="entropy",max_depth = 4) 
drugTree.fit(X_trainset,y_trainset) 
predtree = drugTree.predict(X_testset) 
from sklearn import metrics 
import matplotlib.pyplot as plt 
print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_testset, predtree)) 
#Visualize the data 
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest') 
