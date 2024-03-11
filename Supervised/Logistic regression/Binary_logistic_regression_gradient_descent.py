import numpy as np
class MyLg_regression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        '''
        Binary logistic regression with gradient descent
        
        (lerning_rate, iterations and threshold is 0.5)
        '''
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def gradient(self, X, y):
        n_samples = X.shape[0]
        
        z = ((X @ self.w) + self.b)
        sigm = 1 / (1 + np.exp(-z))
        dw = ((sigm - y).T @ X) * (1 / n_samples)
        db = (sigm - y).sum() * (1 / n_samples)
        return dw, db
    
    def fit(self, X_train, y_train):
        self.w = np.ones(X_train.shape[1])
        self.b = .5
        
        for _ in range(self.iterations):
            dw, db = self.gradient(X_train, y_train)
            self.w -= dw * self.learning_rate
            self.b -= db * self.learning_rate
                
    def predict(self, X):
        prob = X @ self.w + self.b
        prob = 1 / (1 + np.exp(-prob))
        return np.array([1 if ans > .5 else 0 for ans in prob])
    
    def score(self, X, y):
        """For only binary"""
        return np.mean(self.predict(X) == y)

    def __str__(self):
        return "MyLogisticRegression"
