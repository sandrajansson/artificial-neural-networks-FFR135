import numpy as np
import csv

def get_data_set(csv_filename):
    with open(csv_filename) as csvfile:
        data_set = list(csv.reader(csvfile, delimiter=','))

    input_data = np.array([list(map(float, (_input[:-1]))) for _input in data_set])
    target_data = np.array([int(_input[-1]) for _input in data_set])
    return(input_data,target_data)

def init_weights_thresholds(M1,M2):
    # Initialize weight vectors as standard normal distributed and threshold vectors as zeros
    W1 = np.random.normal(0, 1, [M1, 2])
    W2 = np.random.normal(0, 1, [M2, M1]);
    W3 = np.random.normal(0, 1, M2);
    theta1 = np.zeros(M1);
    theta2 = np.zeros(M2);
    theta3 = 0
    return(W1, W2, W3, theta1, theta2, theta3)


input_training, target_training = get_data_set('training_set.csv')
input_validation, target_validation = get_data_set('validation_set.csv')
p_train = len(input_training)
p_val = len(input_validation)

eta = 0.02
max_iterations = 10

M1 = 10
M2 = 6

W1, W2, W3, theta1, theta2, theta3 = init_weights_thresholds(M1,M2)

# Initialize states of hidden layers and output
V1 = np.tanh(-theta1 + np.dot(W1,input_training.transpose()).transpose())
V2 = np.tanh(-theta2 + np.dot(W2,V1.transpose()).transpose())
O = np.tanh(-theta3 + np.dot(W3,V2.transpose()).transpose())

# Use weights on validation set and compute the error
V1_val = np.tanh(-theta1 + np.dot(W1,input_validation.transpose()).transpose())
V2_val = np.tanh(-theta2 + np.dot(W2,V1_val.transpose()).transpose())
O_val = np.tanh(-theta3 + np.dot(W3,V2_val.transpose()).transpose())
C = np.sum(np.absolute(np.sign(O_val) - target_validation))/(2*p_val)

iterations = 0

while C > 0.12 and iterations < max_iterations:

    iterations = iterations + 1

    # Update weights and thresholds according to stochastic gradient descent
    ny = np.random.randint(p_train)

    delta_output = (target_training[ny] - O[ny])*(1 - O[ny]**2)
    delta_hidden2 = delta_output * W3 * (1 - V2[ny]**2)
    delta_hidden1 = np.dot(delta_hidden2,W2) * (1 - V1[ny]**2)

    dW3 = eta*delta_output*V2[ny]
    dW2 = eta*np.outer(delta_hidden2,V1[ny])
    dW1 = eta*np.outer(delta_hidden1,input_training[ny])

    dtheta3 = -eta*delta_output
    dtheta2 = -eta*delta_hidden2
    dtheta1 = -eta*delta_hidden1

    W3 = W3 + dW3
    W2 = W2 + dW2
    W1 = W1 + dW1
    theta3 = theta3 + dtheta3
    theta2 = theta2 + dtheta2
    theta1 = theta1 + dtheta1

    # Update states of hidden layers and output
    V1 = np.tanh(-theta1 + np.dot(W1,input_training.transpose()).transpose())
    V2 = np.tanh(-theta2 + np.dot(W2,V1.transpose()).transpose())
    O = np.tanh(-theta3 + np.dot(W3,V2.transpose()).transpose())

    # Use updated weights and thresholds on validation set and compute the classification error
    V1_val = np.tanh(-theta1 + np.dot(W1,input_validation.transpose()).transpose())
    V2_val = np.tanh(-theta2 + np.dot(W2,V1_val.transpose()).transpose())
    O_val = np.tanh(-theta3 + np.dot(W3,V2_val.transpose()).transpose())
    C = np.sum(np.absolute(np.sign(O_val) - target_validation))/(2*p_val)

print(C)

