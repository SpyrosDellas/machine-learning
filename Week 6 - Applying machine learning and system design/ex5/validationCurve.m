function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%       end

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

# lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 0.7 1 1.5 2 2.5 3 3.5 4 5 6 7 8 9 10]';

# Get the size of lambda_vec
lambdas = length(lambda_vec);

# Initialize the two error vectors
error_train = zeros(lambdas, 1);
error_val = zeros(lambdas, 1);


for i = 1:lambdas
  
  lambda = lambda_vec(i);
  
  # Find optimum theta
  theta = trainLinearReg(X, y, lambda);
  
  # Calculate training set error 
  [error_train(i), grad] = linearRegCostFunction(X, y, theta, 0);
  
  # Calculate cross validation set error 
  [error_val(i), grad] = linearRegCostFunction(Xval, yval, theta, 0);
  
endfor

endfunction