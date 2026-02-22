import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from math import log

e = 2.71828
epoches = 100
lr = 0.005


def relu(x):
    return max(0, x)

def predict(w, x):
    return sum(w_i * x_i for w_i, x_i in zip(w, x))

def to_one_hot(index):
    vector = [0] * 2
    vector[index] = 1
    return vector


def softmax(t):
    exps = []
    for t_i in t:
        exps.append(e ** t_i)
    softmax_vals = [x / sum(exps) for x in exps]

    return softmax_vals


def cross_entropy(z, y, eps=1e-15):
    e = []
    for i in range(len(y)):
        p = min(max(z[i], eps), 1 - eps)  # ограничиваем z[i] в [eps, 1-eps]
        e.append(y[i] * log(p))
    return -sum(e)



X, y = make_classification(
    n_samples=5000,    # количество строк
    n_features=3,     # количество признаков
    n_informative=3,  # количество полезных признаков
    n_redundant=0,     # количество "шумных" признаков
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




weights = []

for i in range(2):
    weights.append(np.random.uniform(0, 1, size=len(X_train[0])))

errors = []
for ep in range(epoches):
    for n, sample in enumerate(X_train):
        y_true = to_one_hot(y_train[n])
        y_preds = [predict(we, sample) for we in weights]
        z = softmax(y_preds)
        loss = cross_entropy(z, y_true)

        for i in range(2):
            weights[i] = weights[i] - lr * sample * (np.array(z) - np.array(y_true))[i]


clear = 0
everyone = 0
for n, sample in enumerate(X_test):
    y_true = to_one_hot(y_test[n])
    y_preds = [predict(we, sample) for we in weights]
    if np.argmax(y_preds) == y_test[n]:
        clear += 1
    everyone += 1

print(clear/everyone)