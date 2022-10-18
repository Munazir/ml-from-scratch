
import numpy as np
from collections import Counter

# imports for model training
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
color_maps = ListedColormap(['red','blue','green'])


def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

class KNN:

    def __init__(self,k=3):
        self.k = k
    
    def fit(self, X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        overall_predictions = [self._get_single_prediction(x) for x in X]
        return overall_predictions

    
    def _get_single_prediction(self,x):

        # compute the distance of give point w.r.t training data
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        # get the closest k indices
        k_closest_indices = np.argsort(distances)[:self.k]

        # get the labels associated with those k indices
        k_labels = [self.y_train[i] for i in k_closest_indices]

        # get the majority votes 
        most_common_label = Counter(k_labels).most_common()

        return most_common_label[0][0]


if __name__ == "__main__":

    # get the training data
    iris_data = datasets.load_iris()
    X, y = iris_data.data , iris_data.target
    print(X.shape)
    print(y.shape)
    
    # plot the data
    plt.figure()
    plt.scatter(X[:,2],X[:,3] , c=y, cmap=color_maps, edgecolors='k',s=20)
    plt.show()

    # get train test split
    X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=1)

    # model training
    knn_clf = KNN(k=3)
    knn_clf.fit(X_train,y_train)

    # get predictions
    predictions = knn_clf.predict(X_test)
    print(predictions)

    # accuracy
    accuracy = np.sum(predictions == y_test)/len(y_test)
    print(accuracy)

    


