function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

# Get the size of the testing dataset X
[m n] = size(X);

# Get the number of labels from Theta2
num_labels = size(Theta2, 1);

# Add the bias parameter, by adding a column of ones to the testing dataset X
X = [ones(m, 1), X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

# Calculate the activations matrix for the hidden layer (layer 1)
# Each row i represents the activations of each neuron for the ith example
A1 = sigmoid(X * Theta1');

# Add a column of ones for the bias parameter
A1 = [ones(m, 1), A1];

# Calculate the activations matrix for the output layer (layer 2)
# Each row i represents the activations the output neurons for the ith example
A2 = sigmoid(A1 * Theta2');

# Find the maximum values and their positions in each row of A2
# By construction, the position of the maximum value corresponds to the label 
# of the output layer neuron with the maximum activation value
[maxValue, p] = max(A2, [], 2);

endfunction