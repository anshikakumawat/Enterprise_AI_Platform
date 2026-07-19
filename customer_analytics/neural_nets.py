import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)  # avoid overflow in exp for large values
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(a):
    return (a > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(a):
    return 1 - a**2

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

act_map = {
    "Sigmoid": (sigmoid, sigmoid_deriv),
    "ReLU": (relu, relu_deriv),
    "Tanh": (tanh, tanh_deriv)
}


class Perceptron:
    # single layer perceptron, only for binary classification
    def __init__(self, n_features, lr=0.1, activation="Sigmoid"):
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 0.5, n_features)
        self.b = 0.0
        self.lr = lr
        self.act, self.act_d = act_map[activation]

    def forward(self, X):
        z = X @ self.w + self.b
        return self.act(z)

    def fit(self, X, y, epochs=50):
        loss_list = []
        acc_list = []
        n = X.shape[0]

        for i in range(epochs):
            out = self.forward(X)
            err = out - y

            dw = X.T @ (err * self.act_d(out)) / n
            db = np.mean(err * self.act_d(out))

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = np.mean(err ** 2)
            pred = (out >= 0.5).astype(int)
            acc = np.mean(pred == y)

            loss_list.append(loss)
            acc_list.append(acc)

        return {"loss": loss_list, "accuracy": acc_list}

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


class MLP:
    # multi layer perceptron with backpropagation
    # layer sizes example [4, 8, 1] 
    #  4 input    1 hidden layer of 8     1 output
    def __init__(self, layer_sizes, activation="ReLU", lr=0.05, task="binary"):
        self.sizes = layer_sizes
        self.lr = lr
        self.task = task
        self.act, self.act_d = act_map[activation]

        rng = np.random.default_rng(42)
        self.W = []
        self.B = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.W.append(w)
            self.B.append(b)

    def forward(self, X):
        a = [X]
        for i in range(len(self.W)):
            z = a[-1] @ self.W[i] + self.B[i]
            last_layer = (i == len(self.W) - 1)
            if last_layer:
                out = softmax(z) if self.task == "multiclass" else sigmoid(z)
            else:
                out = self.act(z)
            a.append(out)
        return a

    def backward(self, a, y):
        n = y.shape[0]
        dW = [None] * len(self.W)
        dB = [None] * len(self.B)

        delta = a[-1] - y  # output error

        for l in reversed(range(len(self.W))):
            dW[l] = a[l].T @ delta / n
            dB[l] = np.mean(delta, axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ self.W[l].T) * self.act_d(a[l])

        return dW, dB

    def fit(self, X, y, epochs=100):
        loss_list = []
        acc_list = []

        for i in range(epochs):
            a = self.forward(X)
            dW, dB = self.backward(a, y)

            for l in range(len(self.W)):
                self.W[l] -= self.lr * dW[l]
                self.B[l] -= self.lr * dB[l]

            out = a[-1]
            eps = 1e-9
            if self.task == "binary":
                loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))
            else:
                loss = -np.mean(np.sum(y * np.log(out + eps), axis=1))

            preds = self.predict(X)
            true = np.argmax(y, axis=1) if self.task == "multiclass" else y.ravel()
            acc = np.mean(preds == true)

            loss_list.append(loss)
            acc_list.append(acc)

        return {"loss": loss_list, "accuracy": acc_list}

    def predict_proba(self, X):
        return self.forward(X)[-1]

    def predict(self, X):
        out = self.predict_proba(X)
        if self.task == "multiclass":
            return np.argmax(out, axis=1)
        return (out.ravel() >= 0.5).astype(int)


if __name__ == "__main__":
    # quick sanity check, run: python neural_engine.py
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                                n_redundant=0, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)

    p = Perceptron(n_features=4, lr=0.1, activation="Sigmoid")
    h1 = p.fit(X, y.ravel(), epochs=50)
    print("perceptron acc:", h1["accuracy"][-1])

    mlp = MLP([4, 8, 1], activation="ReLU", lr=0.1, task="binary")
    h2 = mlp.fit(X, y, epochs=100)
    print("mlp acc:", h2["accuracy"][-1])