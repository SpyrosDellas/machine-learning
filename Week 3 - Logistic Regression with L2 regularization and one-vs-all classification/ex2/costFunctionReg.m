function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;   # Sizes: X(m by n+1), theta(n+1 by 1), z(m by 1)
s = sigmoid(z);  # Size of s is (m by 1)

# Calculate the cost with L2 regularization
n = length(theta);
J = -(1 / m) * (y' * log(s) + (1 - y)' * log(1 - s))... 
    + (lambda / (2*m)) * theta(2:n)' * theta(2:n); #We skip theta(1), i.e. theta0 

# Calculate the gradient corresponding to theta0, without regularisation
grad(1) = -(1 / m) * X'(1, :) * (y - s);

# Calculate the gradient corresponding to theta1...thetan, with regularisation
grad(2:n) = -(1 / m) * X'(2:n, :) * (y - s)...
            + (lambda / m) * theta(2:n);

end