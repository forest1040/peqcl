import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt

n_qubit = 4
dev = qml.device("lightning.qubit", wires=n_qubit)

def layer(v):
    for i in range(n_qubit):
        if i != n_qubit - 1:
            qml.CNOT(wires=[i, i+1])
        qml.RX(v[0], wires=i)
        qml.RY(v[1], wires=i)
        if i != n_qubit - 1:
            qml.CNOT(wires=[i, i+1])
        qml.RY(v[2], wires=i)
        qml.RX(v[3], wires=i)

@qml.qnode(dev)
def quantum_neural_net(var, x):
    # Encode input x into quantum state
    for i in range(n_qubit):
        qml.RY(np.arcsin(x) * 2, wires=i)
        qml.RZ(np.arccos(x * x) * 2, wires=i)

    # "layer" subcircuits
    for v in var:
        layer(v)

    #return qml.expval(qml.QuadX(0))
    #return qml.expval(qml.NumberOperator(0))
    return qml.expval(qml.PauliZ(0))

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cost(var, features, labels):
    preds = [quantum_neural_net(var, x) for x in features]
    return square_loss(labels, preds)

def generate_noisy_sine(x_min, x_max, num_x):
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [np.sin(np.pi * x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    # return np.array(x_train), np.array(y_train)
    return np.array(x_train).flatten(), np.array(y_train)


x_min = -1.0
x_max = 1.0
num_x = 80
X, Y = generate_noisy_sine(x_min, x_max, num_x)
x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)

np.random.seed(0)

# TODO: 2
num_layers = 2
var_init = 0.05 * np.random.randn(num_layers, n_qubit * 4, requires_grad=True)
print(var_init)

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

# TODO: 100
var = var_init
for it in range(100):
    (var, _, _), _cost = opt.step_and_cost(cost, var, X, Y)
    #print("Iter: {:5d} | Cost: {:0.7f} ".format(it, _cost))
    print(f"Iter: {it} | Cost: {_cost}")

predictions = [quantum_neural_net(var, x_) for x_ in x_test]

plt.figure()
plt.scatter(X, Y)
plt.scatter(x_test, predictions, color="green")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
#plt.show()
plt.savefig("sin.png")
