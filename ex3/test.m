%test for ex3_nn.m
Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X)
%p = 
%  4
%  1
%  1
%  4
%  4
%  4
%  4
%  2