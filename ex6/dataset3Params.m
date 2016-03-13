function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
%
steps = [0.01 0.03 0.1 0.3 1 3 10 30];
lowest_mean_err = -1;
for c_val = steps
  for sigma_val = steps
    model = svmTrain(X, y, c_val, @(X1, X2)gaussianKernel(X1, X2, sigma_val)); 
    predictions = svmPredict(model, Xval);
    mean_error = mean(double(predictions ~= yval));
    if lowest_mean_err == -1 || mean_error < lowest_mean_err 
      lowest_mean_err = mean_error;
      C = c_val;
      sigma = sigma_val;
  end
end

% =========================================================================

end
