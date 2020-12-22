function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit


# We will initialize the weights using Xavier initialization, i.e.
# by setting the variance of the weights of each single neuron to 
# (1 / Number of Features) in the previous layer
XavierVariance = 1 / L_in;

# Create the matrix of weights using the normal distribution of mean = 0
# and standard deviation = 1 and then normalize by multiplying each weight 
# by the Xavier standard deviation
# This achieves our target variance for the weights for each neuron
W = sqrt(XavierVariance) .* randn(L_out, L_in);

# Finally, add a column of zeros for bias
W = [zeros(L_out, 1), W];

endfunction