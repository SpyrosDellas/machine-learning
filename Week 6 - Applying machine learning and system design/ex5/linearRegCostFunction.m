function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# Get the number of training examples
m = length(y); 

# Calculate cost without regularization
J = (1 / (2 * m)) * (X * theta - y)' * (X * theta - y);

# Calculata the L2-regularization term, ignoring the bias term theta(1)
L2Reg = (lambda / (2 * m)) * theta(2: end)' * theta(2: end);

J = J + L2Reg;

# Calculate the gradient dJ/dtheta of the cost term without regularization
grad = (1 / m) * X' * (X * theta - y);

# Calculate the gradient dL2Reg/dtheta of the regularization term
# Note: 
# We need to add a zero for the derivative of the bias term to match the 
# dimensions for vector addition
dL2Reg = (lambda / m) * [0; theta(2: end)];

grad = grad + dL2Reg;

endfunction