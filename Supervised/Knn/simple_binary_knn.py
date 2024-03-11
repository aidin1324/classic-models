import numpy as np
class MyKNN:
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        '''
        Works only with numpy array, metric = [euclidean, manhattan]
        weights = [uniform, distance]
        '''
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
    
    @staticmethod
    def euclidean(x, y):
        return np.linalg.norm(x - y, axis=1)
    
    @staticmethod
    def manhattan(x, y):
        return np.sum(abs(x - y))
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X):
        answers = np.array([])
        
        for x in X:
            if self.metric == 'euclidean':
                distances = self.euclidean(x, self.X_train)
            elif self.metric == 'manhattan':
                distances = self.manhattan(x, self.X_train)
            else:
                raise ValueError(f"There is no such as metrics {self.metric}")
            
            n_nearest_index = np.argsort(distances)[:self.n_neighbors]
            n_nearest_label = self.y_train[n_nearest_index]
            
            if self.weights == 'uniform':
                weights = [1] * self.n_neighbors
            elif self.weights == 'distance':
                weights = 1 / distances[n_nearest_index]
                
            most_common = np.bincount(n_nearest_label, weights=weights).argmax()
            answers = np.append(answers, most_common)
        
        return answers
    
    def score(self, X, y):
        """For only binary"""
        return np.mean(self.predict(X) == y)
    
    def __str__(self):
        return "MyKNN"
