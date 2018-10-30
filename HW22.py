import numpy as np

p = 16
eta = 0.02
max_iterations = 10**5

# Inputs
inputs = np.array([[-1,-1,-1,-1],
    [1,-1,-1,-1],
    [-1,1,-1,-1],
    [-1,-1,1,-1],
    [-1,-1,-1,1],
    [1,1,-1,-1],
    [1,-1,1,-1],
    [1,-1,-1,1],
    [-1,1,1,-1],
    [-1,1,-1,1],
    [-1,-1,1,1],
    [1,1,1,-1],
    [1,1,-1,1],
    [1,-1,1,1],
    [-1,1,1,1],
    [1,1,1,1]])

# Targets
A = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1])
B = np.array([-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1])
C = np.array([1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1])
D = np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1])
E = np.array([-1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1])
F = np.array([1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1])

# Choose one target
target = C

# Initialize threshold theta and weights
theta = np.random.uniform(-1,1)    # [-1,1]
weights = np.random.uniform(-0.2,0.2,4)    # [-0.2,0.2]


# Calculate output given initialized threshold and weights
O = np.tanh((np.dot(inputs,weights) - theta)/2)

res = np.array_equal(np.sign(O),target)

# Update weights and threshold according to stochastic gradient descent
# until output is equal to target or maximum number of iterations reached
iterations = 0

while res == False and iterations < max_iterations:

    iterations = iterations + 1

    ny = np.random.randint(p);

    # Update weights according to stochastic gradient descent
    weights = np.add(weights, eta/2*(target[ny]-O[ny])*(1-O[ny]**2)*(inputs[ny]))

    # Update theta according to stochastic gradient descent
    theta = np.add(theta, -eta/2*(target[ny]-O[ny])*(1-O[ny]**2))

    # Update output value given new value of weights and threshold
    O = np.tanh((np.dot(inputs, weights) - theta) / 2)

    res = np.array_equal(np.sign(O), target)


if res == True:
    print('Linearly separable')
else:
    print('Max number of iterations reached, consider trying again before concluding function is not linearly separable')