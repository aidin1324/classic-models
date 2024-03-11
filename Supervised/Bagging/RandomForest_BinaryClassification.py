from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MyRandomForest():
    def __init__(self, n_estimators=100, criteria='gini', min_samples_split=2, max_depth=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        if criteria == 'entropy':
            self.criteria = criteria
        elif criteria == 'gini':
            self.criteria = criteria
        else:
            raise ValueError("Incorrect criteria!")
            
    def bootstraps(self, X, y):
        random_index = np.random.choice(X.shape[0], y.shape[0])
        
        X_sample = X[random_index] 
        y_sample = y[random_index]
        return X_sample, y_sample
        
    def fit(self, X_train, y_train):
        params = {'criterion': self.criteria, 'min_samples_split': self.min_samples_split,
                 'max_depth': self.max_depth, 'max_features': round(np.sqrt(X_train.shape[1]))}
        
        cart = []
        
        for i in range(self.n_estimators):
            X_sample, y_sample = self.bootstraps(X_train, y_train)
            model = DecisionTreeClassifier(**params)
            model.fit(X_sample, y_sample)
            cart.append(model)
            
        self.array_carts = cart
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.n_estimators))
        
        for i in range(self.n_estimators):
            y_pred[:, i] = self.array_carts[i].predict(X)
        
        answers = np.array([])
        
        for row in y_pred:
            answers = np.append(answers, np.argmax(np.bincount(row.astype("int32"))))
        return answers
                       
    def score(self, X, y):
        """For only binary"""
        return np.mean(self.predict(X) == y)
    
    def __str__(self):
        return "MyRandomForest"
