function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%mu = mean(X);
%X_norm = bsxfun(@minus, X, mu);

%sigma = std(X_norm);
%X_norm = bsxfun(@rdivide, X_norm, sigma);

 X_norm = zeros(size(X));
 [m,n] = size(X);
 sigma = ones(1,n);
 mu = mean(X);
 
 for i = 1:m
     X_norm(i,:) = X(i,:) - mu;
 end
 
 %sigma = std(X);
 sigma = std(X_norm);
  
 for j = 1:n
     if mu(j) == 0;
         sigma(j)=1;
     end
 end
 
 for i =1:m
     X_norm(i,:) = X_norm(i,:)./sigma;
 end

% ============================================================

end
