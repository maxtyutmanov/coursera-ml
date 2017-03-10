function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

hyp = sigmoid(theta'*X')';
theta_sq = theta .^ 2;
theta_sq(1) = 0;
theta_pen = lambda / (2 * m) * sum(theta_sq)

J = (1/m) * sum((-y) .* log(hyp) - (1 - y) .* log(1 - hyp)) + theta_pen;

grad_reg = lambda / m * theta;
grad_reg(1) = 0;
er = hyp - y;
grad = (1/m) * (er'*X)' + grad_reg;


% =============================================================

end
