import numpy as np


class CustomOVRC:
    def __init__(self, base_classifier, n_classes):
        factory = BinaryMethodFactory
        self.n_classes = n_classes
        self.classifiers = [
            factory.get_method(base_classifier) for _ in range(self.n_classes)
        ]

    def fit(self, X, y):
        for i in range(self.n_classes):
            y_binary = (y == i).astype(int)
            self.classifiers[i].fit(X, y_binary)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            predictions[:, i] = self.classifiers[i].predict(X)
        return np.argmax(predictions, axis=1)


class BinaryMethodFactory:
    def __init__(self) -> None:
        pass

    def get_method(methodType):
        if methodType == "LogReg":
            return LogisticRegression()
        if methodType == "SVM":
            return SupportVectorMachine()
        else:
            ValueError(methodType)


class LogisticRegression:
    def __init__(self, learning_rate=0.2, n_iters=1500):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return [1 if i > 0.5 else 0 for i in predictions]


class SupportVectorMachine:

    def __init__(self, learning_rate=0.1, lambda_param=0.02, n_iters=1500):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
