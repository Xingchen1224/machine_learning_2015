--------------------------------------------------
>>[theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
% then type in these variable names, to display the results
>>theta
theta =
    5.2148
   -0.5733
>>J_hist(1)
ans  =  5.9794
>>J_hist(1000)
ans = 0.85426
------------------------------------------------
>>[theta J_hist] = gradientDescent([3 5; 1 2; 9 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
>>theta
theta =
    0.2588
    0.3999

----------------------------------------------
>>featureNormalize([1 2 3]')
ans =
  -1
   0
   1

>>featureNormalize([1 2 3;6 4 2]')
ans =
  -1   1
   0   0
   1  -1

>>featureNormalize( [ 8 1 6;  3 5 7; 4 9 2 ] )
ans =
    1.1339   -1.0000    0.3780
   -0.7559         0    0.7559
   -0.3780    1.0000   -1.1339

>>featureNormalize([1 2 3 1;6 4 2 0;11 3 3 9;4 9 8 8]')
ans =
  -0.78335   1.16190   1.09141  -1.46571
   0.26112   0.38730  -0.84887   0.78923
   1.30558  -0.38730  -0.84887   0.33824
  -0.78335  -1.16190   0.60634   0.33824

------------------------------------------------------------
>>computeCostMulti( [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ], [ 2; 5; 5; 6 ], [ 0.3816;  0.7655;  0.7952 ] )
ans =
    6.7273

--------------------------------------------------------------------
>>gradientDescentMulti([3 5 6; 1 2 3; 9 4 2],[1 6 4]',[0 0 0]',0.01,1000)
ans =
   1.2123
  -2.9458
   2.3219

---------------------------------------------------------------------------
>>normalEqn([1 0; 0 2],[1 1]')
ans =
   1.00000
   0.50000