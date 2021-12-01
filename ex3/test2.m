%Test 2a:
%input:
X = sin(1:4)';
y = [1; 2; 2; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda)

%output:
%all_theta =
%
%  -1.83789   1.60864
%  -0.27048   0.91450
%  -1.02833  -2.76145