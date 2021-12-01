function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%x_sz = size(X)
%theta1_sz = size(Theta1)
%theta2_sz = size(Theta2)
%p_sz = size(p)


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

% return p(predictions) which is the index vector of maximum value in each example (row) in a3
[values_max , index_max] = max(a3,[],2);
p = index_max;
% =========================================================================


end
