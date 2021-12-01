%Test 5a (Regularized Gradient):
%input:
input_layer_size = 2;
hidden_layer_size = 2;
num_labels = 4;
nn_params = [ 1:18 ] / 10;
X = cos([1  2 ; 3  4 ; 5  6]);
y = [4; 2; 3];
lambda = 3;
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%output:
%J =  16.457
%grad =

%   0.76614
%   0.97990
%   0.27246
%   0.36416
%   0.47507
%   0.54614
%   0.88342
%   0.56876
%   0.58467
%   0.59814
%   1.55931
%   1.54462
%   1.55631
%   1.71189
%   1.97834
%   1.96892
%   1.95977
%   2.12233