function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
# Implements the neural network 'L2' regularised cost function and its 
# derivatives for a two layer neural network (1 input, 1 hidden and 1 output).
# The network performs classification of the MNIST dataset hand written digit 
# images. Each image is 20 by 20 pixel, grayscale.
# 
# Inputs:
# - vector nn_params:         The parameters (weights) for the neural network 
#                             unrolled (flattened) into a vector
# - scalar input_layer_size:  The size of the input layer  
# - scalar hidden_layer_size: The size of the hidden layer 
# - scalar num_labels:        The size of the output layer; equal to the number
#                             of classes in the dataset
# - matrix X:                 The training dataset features; each row represents
#                             a single training example
# - vector y:                 The training dataset labels, ranging from
#                             1..num_labels
# - scalar lambda:            The regularization parameter        
#
# Outputs:
# - scalar J:    The cost function of the neural network
# - vector grad: An unrolled (flattened) vector of the partial 
#                derivatives of the neural network


% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.

#-------------------------------------------------------------------------------
# INPUTS INITIALIZATION

# Reshape nn_params back into the weight matrices Theta1 and Theta2
split = hidden_layer_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1: split), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + split): end), ...
                 num_labels, (hidden_layer_size + 1));

# Get dimensions of X
[m, n] = size(X);

# Add the bias parameter, by adding a column of ones to X
X = [ones(m, 1), X];

# Convert y to a matrix of size num_labels by m, where the ith column represents
# the class of the ith training example in the form [0..0,1,0..0]
classes = [1: num_labels];
y_train = (y(1: end) == classes)';  # y_train (num_labels by m)

#-------------------------------------------------------------------------------
# LAYER 1 (HIDDEN LAYER) FORWARD PROPAGATION - FULLY VECTORISED IMPLEMENTATION

# Calculate matrices Z1 and A1, where each column i represents the activations 
# of each neuron for the ith example
Z1 = Theta1 * X';  # Theta1 (hidden_layer_size by n+1), X' (n+1 by m)
A1 = sigmoid(Z1);  # A1 (hidden_layer_size by m)

# Add a row of ones for the bias parameter
A1 = [ones(1, m); A1];

#-------------------------------------------------------------------------------
# LAYER 2 (OUTPUT LAYER) FORWARD PROPAGATION - FULLY VECTORISED IMPLEMENTATION

# Calculate matrices Z2 and A2, where each column i represents the activations 
# of each neuron for the ith example
Z2 = Theta2 * A1;  # Theta2 (num_labels by hidden_layer_size + 1), 
                   # A1 (hidden_layer_size +1 by m)
A2 = sigmoid(Z2);  # A2 (num_labels by m)

#-------------------------------------------------------------------------------
# COST CALCULATION - FULLY VECTORISED IMPLEMENTATION

# Calculate losses for each class and each training example
losses = y_train .* log(A2) + (1 - y_train) .* log(1 - A2);

# Calculate cost by summing first along the classes axis and then along
# the training examples axis
J = -(1 / m) * sum(sum(losses, 1));

# Calculate the L2-regularization term
# Note 1: First calculate the squared Frobenius norms of the weight matrices. 
# Note 2: The weights for the bias parameter are excluded
squaredNorm1 = sum(sum(Theta1(:, 2: end) .^ 2, 1));
squaredNorm2 = sum(sum(Theta2(:, 2: end) .^ 2, 1));
L2RegTerm = (lambda / (2 * m)) * (squaredNorm1 + squaredNorm2);

J = J + L2RegTerm;

#-------------------------------------------------------------------------------
# LAYER 2 (OUTPUT LAYER) BACK PROPAGATION - FULLY VECTORISED IMPLEMENTATION

# Calculate the partial derivatives of the cost with respect to Z2 
# dJ/dZ2 = -(1/m) * (y - A2)
dZ2 = -(1 / m) * (y_train - A2); # y_train(num_labels by m),
                                 # A2(num_labels by m),
                                 # dZ2 (num_labels by m)

# Calculate the partial derivatives of the cost with respect to Theta2
# dJ/dTheta2 = dJ/dZ2 * dZ2/dTheta2 = dZ2 * A1'            
Theta2_grad = dZ2 * A1';   # dZ2 (num_labels by m)
                           # A1' (m by hidden_layer_size +1)
                           # Theta2_grad (num_labels by hidden_layer_size + 1)

# Calculate the gradient of the L2-regularization term with respect to Theta2
L2Grad2 = (lambda / m) * [zeros(num_labels, 1), Theta2(:, 2: end)];

Theta2_grad = Theta2_grad + L2Grad2;
                        
#-------------------------------------------------------------------------------
# LAYER 1 (HIDDEN LAYER) BACK PROPAGATION - FULLY VECTORISED IMPLEMENTATION

# Calculate the partial derivatives of the cost with respect to A1 
# dJ/dA1 = dJ/dZ2 * dZ2/dA1 = dZ2 * Theta2 (since Z2 = Theta2 * A1)
dA1 = Theta2' * dZ2;  # Theta2' (hidden_layer_size + 1 by num_labels),
                      # dZ2 (num_labels by m)
                      # dA1 (hidden_layer_size + 1 by m)
                                 
# Calculate the partial derivatives of the cost with respect to Z1
# dJ/dZ1 = dJ/dA1 * dA1/dZ1  = dA1 * [sigmoid(Z1) .* (1 - sigmoid(Z1)] 
#
# Here we avoid using the sigmoidGradient function implemented for the
# course auto grader, as it would result calculating again the sigmoid(Z1),
# which is already stored in A1
# THIS IS SLOW!!! sGrad = sigmoidGradient(Z1); 
#
# We ignore the first row of ones corresponding to the bias parameter
sGrad = A1(2: end, :) .* (1 - A1(2: end, :));
dZ1 = dA1(2: end, :) .* sGrad;   # dA1(2:end,:) (hidden_layer_size by m)
                                 # sGrad (hidden_layer_size by m) 
                                 # dZ1 (hidden_layer_size by m)

# Calculate the partial derivatives of the cost with respect to Theta1
# dJ/dTheta1 = dJ/dZ1 * dZ1/dTheta1 = dZ1 * X'
Theta1_grad = dZ1 * X;  # dZ1 (hidden_layer_size by m)
                        # X (m by input_layer_size + 1)
                        # Theta1_grad (hidden_layer_size by input_layer_size + 1)

# Calculate the gradient of the L2-regularization term with respect to Theta1
L2Grad1 = (lambda / m) * [zeros(hidden_layer_size, 1), Theta1(:, 2: end)];

Theta1_grad = Theta1_grad + L2Grad1;
                        
# Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];

endfunction