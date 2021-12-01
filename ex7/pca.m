function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

%X(isnan(X)) = 0 ;

[U,S,V] = svd(X'*X/m);

overAllSum = sum(sum(S));

for i = 1:n
   cof = sum(sum(S(1:i,1:i)))/overAllSum;
   if cof > 0.99
       fprintf('We need %d features to make it more than 99%', i);
       break;
   end
end


% =========================================================================

end
