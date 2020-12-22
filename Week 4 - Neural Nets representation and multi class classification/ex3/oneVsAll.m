function [all_theta] = oneVsAll(X, y, num_labels, lambda)
# [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
# "L2 regularised" logistic regression classifiers
#
# Inputs:
# - matrix X: the training dataset of m training examples by n features each
# - vector y: vector of m labels, each label ranging from 1 to K, where 
#             K is the number of classes in our training dataset 
# - scalar lambda: the regularisation parameter
#
# Outputs:
# - matrix all_theta: the weights of each classifier, where the i-th row 
#                     corresponds to the classifier for label i

% ====================== YOUR CODE HERE ======================
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);

# Get the size of the training dataset X
[m n] = size(X);

# Add the bias parameter, by adding a column of ones to the X data matrix
X = [ones(m, 1) X];

# Initialize all_theta to zeros
all_theta = zeros(num_labels, n + 1);

# Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

for label = 1:num_labels
  # Prepare training labels vector
  yTrain = (y == label);
  # Run fmincg to obtain the optimal theta
  [all_theta(label, :)] = fmincg (@(t)(lrCostFunction(t, X, yTrain, lambda)), ...
                    all_theta(label, :)', options);
endfor


endfunction