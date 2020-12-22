function [jVal, gradient] = costFunction(theta)
  
  % Calculate the function value
  jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;
  
  % Calculate the function gradient
  gradient = zeros(2,1);
  gradient(1) = 2 * (theta(1) - 5);
  gradient(2) = 2 * (theta(2) - 5);
  
endfunction