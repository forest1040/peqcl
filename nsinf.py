import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt

dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

def layer(v):
    # Matrix multiplication of input layer
    qml.Rotation(v[0], wires=0)
    qml.Squeezing(v[1], 0.0, wires=0)
    qml.Rotation(v[2], wires=0)

    # Bias
    qml.Displacement(v[3], 0.0, wires=0)

    # Element-wise nonlinear transformation
    qml.Kerr(v[4], wires=0)

@qml.qnode(dev)
def quantum_neural_net(var, x):
    # Encode input x into quantum state
    qml.Displacement(x, 0.0, wires=0)

    # "layer" subcircuits
    for v in var:
        layer(v)

    #return qml.expval(qml.QuadX(0))
    #return qml.expval(qml.PauliZ(0))
    return qml.expval(qml.NumberOperator(0))

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cost(var, features, labels):
    preds = [quantum_neural_net(var, x) for x in features]
    return square_loss(labels, preds)


# TODO: noisy sin

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

#data = np.loadtxt("sine.txt")
#X = np.array(data[:, 0], requires_grad=False)
#Y = np.array(data[:, 1], requires_grad=False)



# plt.figure()
# plt.scatter(X, Y)
# plt.xlabel("x", fontsize=18)
# plt.ylabel("f(x)", fontsize=18)
# plt.tick_params(axis="both", which="major", labelsize=16)
# plt.tick_params(axis="both", which="minor", labelsize=16)
# plt.show()

np.random.seed(0)

# TODO: 4
num_layers = 4
var_init = 0.05 * np.random.randn(num_layers, 5, requires_grad=True)
print(var_init)

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

# TODO: 500
var = var_init
for it in range(500):
    (var, _, _), _cost = opt.step_and_cost(cost, var, X, Y)
    #print("Iter: {:5d} | Cost: {:0.7f} ".format(it, _cost))
    print(f"Iter: {it} | Cost: {_cost}")

# Finally, we collect the predictions of the trained model for 50 values
# in the range :math:`[-1,1]`:

# x_pred = np.linspace(-1, 1, 50)
# predictions = [quantum_neural_net(var, x_) for x_ in x_pred]
predictions = [quantum_neural_net(var, x_) for x_ in x_test]


plt.figure()
plt.scatter(X, Y)
plt.scatter(x_test, predictions, color="green")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
#plt.show()
plt.savefig("sine.png")

# variance = 1.0

# plt.figure()
# x_pred = np.linspace(-2, 2, 50)
# for i in range(7):
#     rnd_var = variance * np.random.randn(num_layers, 7)
#     predictions = [quantum_neural_net(rnd_var, x_) for x_ in x_pred]
#     plt.plot(x_pred, predictions, color="black")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.tick_params(axis="both", which="major")
# plt.tick_params(axis="both", which="minor")
# plt.show()

