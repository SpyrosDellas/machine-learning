function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;   # Sizes: X(m by n+1), theta(n+1 by 1), z(m by 1)
s = sigmoid(z);  # Size of s is (m by 1)

# Calculate the cost
J = -(1 / m) * (y' * log(s) + (1 - y)' * log(1 - s))

# Calculate the gradient
grad = -(1 / m) * X' * (y - s);

end