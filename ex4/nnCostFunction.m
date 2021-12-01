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

%forward propagation algorithm

%add ones to X
%X = [ones(m, 1) X];

%predictions: h_x
	a1 = X;
	%add a_0^1
	vec_ones_4_input = ones(size(a1,1),1);
	a1_prim = [vec_ones_4_input a1];

	%computer z2 , a2
	z2 = a1_prim * Theta1';
	a2 = sigmoid(z2);

	%add a_0^2
	vec_ones_4_1st_hidden_layer = ones(size(a2),1);
	a2_prim = [vec_ones_4_1st_hidden_layer a2];

	%compute z3 , a3
	z3 = a2_prim * Theta2';
	a3 = sigmoid(z3);
	
	h_x = a3;

%X_sz = size(X)
%y_sz = size(y)
%h_x_sz = size(h_x)

%mapping y = 2 to y = [0,1,0,0,...,0,0]
y_prim = zeros(m,num_labels);
	for i = 1:m
		y_prim(i,y(i)) = 1;
	end

%un-regularized cost function
J = sum(sum(-y_prim.*log(h_x) - (1 - y_prim).*log(1 - h_x))) / m;

%regularized cost function
J = J + lambda * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) / (2*m);


%backpropagation algorithm
%un-regularized gradient

for t = 1:m
	%step 1: set a_1 -> t_th training example, add bias unit (1)
	a_1 = X(t,:);
	a_1 = [1  a_1];
	%compute z_2, a_2, z_3, a_3
	z_2 = a_1 * Theta1';
	a_2 = sigmoid(z_2);
	a_2 = [1 a_2];
	z_3 = a_2 * Theta2';
	a_3 = sigmoid(z_3);
	
	%step 2: 
	delta_3 = a_3 - y_prim(t,:); %y_prim was defined in forward propagation	
	
	%step 3:
	delta_2 = delta_3*Theta2;
	delta_2 = delta_2(2:end).*sigmoidGradient(z_2);
	
	%step 4:
	%sz_delta_3 = size(delta_3)
	%sz_delta_2 = size(delta_2)
	
	Theta2_grad = Theta2_grad + delta_3'*a_2;
	Theta1_grad = Theta1_grad + delta_2'*a_1;
		
end

%step 5:
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

%regularized gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m; 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
