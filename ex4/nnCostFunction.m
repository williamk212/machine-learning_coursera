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
X1 = [ones(m, 1) X];

% calculate hidden layer
z2 = Theta1 * X1';
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
% printf("size hypo: %ix%i\n", hm, hn);
% printf("sample hypo: %i\n", hypo(2, :));

% convert y to y_mat
y_eye = eye(num_labels);
y_mat = y_eye(y, :);
[ymatm, ymatn] = size(y_mat);
% printf("size y-matrix: %ix%i\n", ymatm, ymatn);

% backpropagation: delta3
d3 = hypo - y_mat;

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

% backpropagation: delta2
% more info: https://www.coursera.org/learn/machine-learning/discussions/i2u9QezvEeSQaSIACtiO2Q/replies/XpcX6-0PEeS0tyIAC9RBcw
[z2m, z2n] = size(z2);
% size(z2): 25x5000
% printf("size z2: %ix%i\n", z2m, z2n);

[d3m, d3n] = size(d3);
% printf("size d3: %ix%i\n", d3m, d3n);

d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2)';
[d2m, d2n] = size(d2);
% printf("size d2: %ix%i\n", d2m, d2n);
%a2 = sigmoid(z2);
[a2m, a2n] = size(a2);
% printf("size a2: %ix%i\n", a2m, a2n);

% NOTES about dimension:
% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications

% delta1 = product d2 and a1 (h x m) * (m x n) --> (h x n)
% a1 = X (w/o 1s), size: 5000x400
% d2, size: 5000x25
delta1 = d2' * X1;
[dt1m, dt1n] = size(delta1);
% printf("size delta1: %ix%i\n", dt1m, dt1n);
% delta2 = product d3 and a2, (r x m) * (m x [h + 1]) --> (r x [h+1])
% d3, size: 5000x10
delta2 = d3' * a2';
% delta2 = delta2(2:end);
[dt2m, dt2n] = size(delta2);
% printf("size delta2: %ix%i\n", dt2m, dt2n);

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;

% add regularized gradient
% note that we should NOT be regularizing the first column of Theta
theta1m = size(Theta1)(1);
theta1_reg = (lambda / m) * ([zeros(theta1m, 1) Theta1(:, 2:end)]);
theta2m = size(Theta2)(1);
theta2_reg = (lambda / m) * ([zeros(theta2m, 1) Theta2(:, 2:end)]);

Theta1_grad = Theta1_grad + theta1_reg;
Theta2_grad = Theta2_grad + theta2_reg;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
