function p = predictOneVsAll(all_theta, X)
# Predicts the labels for a trained one-vs-all classifier. 
# The labels are in the range 1..K, where K = size(all_theta, 1)
#
# Inputs:
# - matrix all_theta: the weights of each trained classifier, where the i-th row 
#                     corresponds to the classifier for label i
# - matrix X: the testing dataset of m testing examples by n features each
#
# Outputs:
# - vector p: predicted label values from 1...K for each testing example,
#             e.g. p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2 for 4 examples 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row

# Get the size of the testing dataset X
[m n] = size(X);

# Get the number of labels from all_theta
num_labels = size(all_theta, 1);

# Add the bias parameter, by adding a column of ones to the testing dataset X
X = [ones(m, 1) X];

# Create a prediction activations matrix of size m by num_labels, where
# each row i contains the activation values of the different classifiers
predictions = sigmoid(X * all_theta');

# Find the maximum values and their positions in each row of predictions
# By construction, the position of the maximum value corresponds to the label 
# of the classifier with the maximum activation value
[maxValue, p] = max(predictions, [], 2);

endfunction