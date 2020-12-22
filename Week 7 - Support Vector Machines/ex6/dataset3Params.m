function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

# Define a range of possible values for C and sigma
paramRange = [0.01 0.03 0.1 0.3 1 3 10 30];

# Set cross validation set prediction error to maximum
valError = 1.0;

for C_ = paramRange
  for sigma_ = paramRange
    # Train SVM using the training set
    model = svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_));
    # Calculate the cross validation set predictions
    predictions = svmPredict(model, Xval);
    # Calculate the prediction error and update C and sigma
    valError_ = mean(double(predictions ~= yval));
    if valError_ <= valError
      valError = valError_
      C = C_ 
      sigma = sigma_
    endif
  endfor
endfor

endfunction