function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

# First find the indices of the ones and zeros
onesIndex = find(y > 0.5);
zerosIndex = find(y < 0.5);

scatter(X(onesIndex, 1), X(onesIndex, 2), 'k', '+', 'linewidth', 1.5);
scatter(X(zerosIndex, 1), X(zerosIndex, 2), 'y', 'o', 'filled', 'markeredgecolor', 'k');

legend('Admitted', 'Not Admitted');
xlabel('Exam 1 score');
ylabel('Exam 2 score');

hold off;

end