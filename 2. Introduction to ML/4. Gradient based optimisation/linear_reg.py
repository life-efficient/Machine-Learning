# %%
from sklearn import datasets
import numpy as np

X, y = datasets.load_boston(return_X_y=True)

print(X.shape)
print(y.shape)
# %%


class LinearRegression:
    def __init__(self, n_features) -> None:
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def fit(self, X, y, epochs=10):
        lr = 0.1
        for epoch in range(epochs): # for a certain number of runs through the whole dataset (epoch)
            # batch our examples
            # for batch in batches
            pred = self.predict(X) # make predictions
            print(pred.shape)
            sdsd
            loss = self._get_mean_squared_error_loss(pred, y) # compute loss and print    
            print('Loss:', loss)
            grad_w, grad_b = self._compute_grads(X, y) # compute gradient for each weight and bias
            self.w -= lr * grad_w # update weight
            self.b -= lr * grad_b # update bias

    def predict(self, X):
        return np.matmul(X, self.w) + self.b

    def _compute_grads(self, X, y):
        y_hat = self.predict(X)
        grad_b = 2 * np.mean(y_hat - y)
        grad_w = 2 * np.mean((y_hat - y)*X)
        return grad_w, grad_b

    def _get_mean_squared_error_loss(self, y_hat, y):
        return np.mean((y_hat - y)**2)

    
# %%
linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X, y)
linear_model.predict(X)
        
# %%
