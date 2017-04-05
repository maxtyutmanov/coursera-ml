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

a1 = X';
a1 = [ones(1, size(a1, 2)); a1];
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% predictions matrix. columns correspond to different training 
% examples, j-th row in i-th example/column represents a kind of
% "probability" that this example should be categorized as j-th label

pred = a3;

% yt matrix is transformed matrix of expected results y in training set
% each column contains single 1 value and other values are set to zero.
% the value of 1 in j-th item (j-th row) of i-th column means that i-th example should be marked with j-th label

yt = zeros(num_labels, m);
for i=1:m
  yt(y(i), i) = 1;
endfor

J = 1 / m * sum(sum((-yt) .* log(pred) - (1 - yt) .* log(1 - pred)));

Theta1_reg = Theta1(:, 2:end) .^ 2;
Theta2_reg = Theta2(:, 2:end) .^ 2;

reg_term = lambda / (2 * m) * (sum(sum(Theta1_reg)) ...
+ sum(sum(Theta2_reg)));

J += reg_term;

% -------------------------------------------------------------

% backpropagation algorithm

% |Theta1| = 25 x 401 - maps 401 items (including bias unit) into
% 25 items of hidden layer
delta1 = zeros(size(Theta1, 1), size(Theta1, 2));
% |Theta2| = 10 x 26 - maps 26 items (including bias unit) into
% 10 items of hidden layer
delta2 = zeros(size(Theta2, 1), size(Theta2, 2));

for t=1:m

  % feedforward using current values of parameter matrices Theta*
  % |a1| = 401
  a1 = X(t,:)';
  a1 = [1; a1];
  z2 = Theta1 * a1;
  % |a2| = 26
  a2 = sigmoid(z2);
  a2 = [1; a2];
  % |a3| = 10
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % calculate delta vectors
  % |d3| = 10 x 1
  d3 = a3 - yt(:,t);
  % |d2| = 25 x 1 (identical to a2)
  d2 = Theta2' * d3 .* (a2 .* (1 - a2));
  d2 = d2(2:end);

  delta1 += d2 * a1';
  delta2 += d3 * a2';

endfor

Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;

Theta1_grad_reg = lambda / m * Theta1;
Theta1_grad_reg(:, 1) = zeros(size(Theta1, 1), 1);
Theta2_grad_reg = lambda / m * Theta2;
Theta2_grad_reg(:, 1) = zeros(size(Theta2, 1), 1);

Theta1_grad += Theta1_grad_reg;
Theta2_grad += Theta2_grad_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
