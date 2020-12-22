% Clear all variables
clear; 
% Close all figure windows
close all; 
% Clear the terminal screen and move cursor to upper left corner
clc

% Create an option structure for the optimisation function:
%
% 1. 'GradObj' = 'on', meaning that the funtion to be optimised returns a second
%                      argument, which is its gradient
% 2. 'MaxIter' = 100, the max number of iterations before optimisation stops
%
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Create the starting point for the optimisation algorithm
initialTheta = zeros(2,1);

% Optimise using 'fminunc'
[optimalTheta, functionVal, exitFlag] = ...
                                   fminunc(@costFunction, initialTheta, options)