#not optimized
#works with numpy array
class Question:
    def __init__(self, column, condition):
        self.column = column
        self.condition = condition
        
    def match(self, data):
        value = data[self.column]
        if len(np.unique(value)) > 2:
            return value >= self.condition
        else:
            return value == self.condition
    
    def __repr__(self):
        return f'{self.condition} {self.column}'
    
class DecisionNode:
    def __init__(self, question, left_node, right_node):
        self.question = question
        self.left_node = left_node
        self.right_node = right_node
        self.root = None
class LeafNode:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class

class Criteria_Mixin:
    def entropy(self, p):
        return -p * np.log2(p) - (1 - p) * np.log2(p)
    
    def gini_impurity(self, p):
        return 1 - p ** 2 - (1 - p) ** 2
    
    def information_gain(self, p, w_left_child, w_right_child, w_parent):
        return w_parent - (p * w_left_child + (1 - p) * w_right_child)
    
    def find_p(self, class_1, total):
        return class_1 / total
    
        

class MyDecisionTreeClassifier(Criteria_Mixin):
    def __init__(self, max_depth=None, max_features=None,
                 min_samples_split=2, criteria='gini'):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        
        if criteria == 'entropy':
            self.criteria = self.entropy
        elif criteria == 'gini':
            self.criteria = self.gini_impurity
        else:
            raise ValueError("Incorrect criteria!")
            
    def particion(self, rows, question):
        true_rows, false_rows = [], []
        
        for index, row in enumerate(rows):
            if question.match(row):
                true_rows.append(index)
            else:
                false_rows.append(index)
        return true_rows, false_rows
    
    def find_best_split(self, X_train, y_train):
        
        n_samples, n_features = X_train.shape
        current_p = self.find_p(y_train[y_train == 1].shape[0], n_samples)
        current_gain = self.criteria(current_p)
        
        best_gain = -1
        best_question = None
      
        for col in range(n_features):
            values = np.unique(X_train[:, col])
        
            for val in values:
                question = Question(col, val)
                
                true_rows, false_rows = self.particion(X_train, question)
                
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                p = len(true_rows) / n_samples
                tr, fr = y_train[true_rows], y_train[false_rows]
                
                w_left = self.criteria(self.find_p(np.sum(tr == 1), len(tr)))
                w_right = self.criteria(self.find_p(np.sum(fr == 1), len(fr)))
                gain = self.information_gain(p, w_left, w_right, current_gain)

                if gain > best_gain:
                    best_gain, best_question = gain, question
        if best_question == None:     
            best_gain = 1 #same features and can not split properly
            
        return best_gain, best_question     
        
    def fit(self, X_train, y_train):
        def recursion_fit(X, y, depth=0):
            n_samples, n_features = X.shape

            if n_samples < self.min_samples_split:
                return LeafNode(predicted_class=y[0])
            rand_features = None
            if self.max_features is not None and self.max_features > 0:
                rand_features = rand_features = np.random.choice(
                    range(n_features), size=self.max_features, replace=False
                )
            
            if rand_features is not None:
                info, question = self.find_best_split(X[rand_features], 
                                                  y[rand_features])
            else:
                info, question = self.find_best_split(X, y)
        
            if info in [1, 0] or (self.max_depth is not None and depth == self.max_depth):
                return LeafNode(predicted_class=np.argmax(np.bincount(y)))
            
            true_rows, false_rows = self.particion(X, question)
            
            true_branch = recursion_fit(X[true_rows], 
                                             y[true_rows], depth+1)
            false_branch = recursion_fit(X[false_rows], 
                                              y[false_rows], depth+1)

            return DecisionNode(question, true_branch, false_branch)
        
        self.root = recursion_fit(X_train, y_train)
    
    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])
    
    def _predict(self, x, node):
        if isinstance(node, LeafNode):
            return node.predicted_class
        elif node.question.match(x):
            return self._predict(x, node.left_node)
        else:
            return self._predict(x, node.right_node)
    
    def score(self, X, y):
        """For only binary"""
        return np.mean(self.predict(X) == y)
    
    def __str__(self):
        return "MyDecisionTreeClassifier"
