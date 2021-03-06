Test 1a (Regularized Linear Regression Cost Function):
input:
J = linearRegCostFunction( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2], 7 )

output:
J =  11.980


---------------------------------------
Test 2a (Regularized Linear Regression Gradient):
input:
[J, grad] = linearRegCostFunction( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2], 7 )

output:
J =  11.980
grad =

   -4.7000
  -14.6000

----------------------
Test 2b (linearRegCostFunction() with ONE training example)
(your function must pass this test for "learningCurve()" to pass the submit grader)
[c g] = linearRegCostFunction([1 -1 1],[1],[2 -2 0]',1)
c = 6.5
g = 
   3
  -5
   3
---------------------
Test 2c (CTA add: uses additional features) - non-reg and reg linearRegCostFunction()
X = [[1 1 1]' magic(3)];
y = [7 6 5]';
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, ?)
--- results based on value entered for ? (lambda)
--------------------------
lambda = 0  |   lambda = 7
--------------------------
J = 1.3533  |   J = 1.6917
g =         |   g = 
   -1.4000  |      -1.4000
   -8.7333  |      -8.2667
   -4.3333  |      -3.6333
   -7.9333  |      -7.0000



---------------------------------------
Test 3a (Learning Curve):
input:
[error_train, error_val] = learningCurve( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [1 7; 1 -2;], [2; 12], 7 )

output:
error_train =

   0.00000
   0.10889
   0.20165
   0.21267

error_val =

   12.5000
   11.1700
    8.3951
    5.4696

Note: this also outputs a bunch of iteration/cost lines, and warnings about division by zero, like the following:
warning: division by zero
Iteration     8 | Cost: 3.645833e-01


---------------------------------------
Test 4a (Polynomial Feature Mapping):
input:
polyFeatures([1:3]', 4)

output:
ans =

    1    1    1    1
    2    4    8   16
    3    9   27   81


---------------------------------------
Test 5a (Validation Curve):
input:
[lambda_vec, error_train, error_val] = validationCurve( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [1 7; 1 -2;], [2; 12] )

output:
lambda_vec =

    0.00000
    0.00100
    0.00300
    0.01000
    0.03000
    0.10000
    0.30000
    1.00000
    3.00000
   10.00000

error_train =

   9.8608e-32
   2.4990e-08
   2.2473e-07
   2.4900e-06
   2.2232e-05
   2.4029e-04
   2.0025e-03
   1.7361e-02
   8.7891e-02
   2.7778e-01

error_val =

   0.25000
   0.25055
   0.25165
   0.25553
   0.26678
   0.30801
   0.43970
   1.00347
   2.77539
   6.80556