%Test 2a (Regularized Cost Function):
%input:
input_layer_size = 2;
hidden_layer_size = 2;
num_labels = 4;
nn_params = [ 1:18 ] / 10;
X = cos([1  2 ; 3  4 ; 5  6]);
y = [4; 2; 3];
lambda = 3;
nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%output:
%ans =  16.457
