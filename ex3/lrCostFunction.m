function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
  [theta_m, theta_n] = size(theta);
  % printf("size theta: %ix%i\n", theta_m, theta_n);
  [x_m, x_n] = size(X);
  % printf("size X: %ix%i\n", x_m, x_n);
  hypo = sigmoid(X * theta);
  [hypo_m, hypo_n] = size(hypo);
  % printf("size hypo: %ix%i\n", hypo_m, hypo_n);
  pos_log = (-1 .* y) .* log(hypo);
  neg_log = (1 .- y) .* log(1 .- hypo);

  sq_term = theta([2:theta_m], :) .^ 2;
  %sq_term = theta .^ 2;
  reg_term = sum(sq_term) * lambda / (2 * m)
  % calculate cost function
  J = (1 / m) * sum(pos_log - neg_log) + reg_term;

  grad = (1 / m) * (X' * (hypo - y));
  theta_no0 = theta(2:end);
  grad_reg = theta_no0 .* lambda ./ m;
  grad = grad + [0; grad_reg];

% =============================================================

grad = grad(:);

end
