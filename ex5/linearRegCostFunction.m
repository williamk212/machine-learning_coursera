function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypo_diff_sq = sum(((X * theta) - y) .^ 2);
[hm, hn] = size(hypo_diff_sq);
% printf("size hypo_diff_sq: %ix%i\n", hm, hn);

J = ( (1 / (2 * m)) * hypo_diff_sq) + ...
    ( lambda / (2 * m) * sum(theta(2:end, :) .^2 ) );

% =========================================================================

grad = grad(:);
hypo = X * theta .- y;
[hm, hn] = size(hypo);
% printf("size hypo: %ix%i\n", hm, hn);
deriv = sum(hypo .* X) / m;

reg_term = theta .* (lambda / m);
reg_term(1) = 0;

grad = deriv' + reg_term;
end
