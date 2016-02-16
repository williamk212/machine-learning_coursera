function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% function signature:
% function [J grad] = nnCostFunction(nn_params, ...
%                                   input_layer_size, ...
%                                   hidden_layer_size, ...
%                                   num_labels, ...
%                                   X, y, lambda)

% let's calculate the hypothesis
% initialize input layer, add 1 to X matrix 
m = size(X, 1);
X = [ones(m, 1) X];

% calculate hidden layer
z2 = Theta1 * X';
a2 = sigmoid(z2);
n_a2 = size(a2, 2);
% initialize hidden layer by adding 1 to matix
a2 = [ones(1, n_a2); a2];
[a2m, a2n] = size(a2);
%printf("size a2 after 1s: %ix%i\n", a2m, a2n);

% calculate output layer
z3 = Theta2 * a2;
hypo = sigmoid(z3)';
[hm, hn] = size(hypo);
%printf("size hypo: %ix%i\n", hm, hn);
% printf("sample hypo: %i\n", hypo(2, :));

% convert y to y_mat
y_eye = eye(num_labels);
y_mat = y_eye(y, :);
% y = 1, calculation
pos_class = (-1 .* y_mat) .* log(hypo);
[loghm, loghn] = size(log(hypo));
%printf("size log hypo: %ix%i\n", loghm, loghn);

% y = 0, calculation
neg_class = (1 .- y_mat) .* log(1 .- hypo); 

J = (1 / m ) .* sum(sum(pos_class - neg_class));

% let's build regularized cost function term
[t1m, t1n] = size(Theta1);
[t2m, t2n] = size(Theta2);

theta1_nobias = Theta1(:, 2:t1n);
theta2_nobias = Theta2(:, 2:t2n);

sum_squares = sum(sum(theta1_nobias .^ 2)) + sum(sum(theta2_nobias .^ 2));
reg_term = (lambda / (2 * m)) .* sum_squares;
J = J + reg_term;
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
