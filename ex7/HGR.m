%% =============== Part 4: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
%fprintf('\nLoading face dataset.\n\n');
clear;
clc;

%  Load Face dataset
%load ('features.mat')

%img hist equlisation
%load ('features-histeq.mat')

%img uint16 blue (2nd channel) left shift 8 + green (3rd channel)
load ('features-color.mat')

figure;
% Display the first 100 faces in the dataset
subplot(1, 2, 1);
displayData(dataMatrix(1:100, :));
title('First 100 gestures');
axis square;


%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
%fprintf(['\nRunning PCA on face dataset.\n' ...
         %'(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize dataMatrix by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(dataMatrix);

%  Run PCA
[U, S] = pca(X_norm);


%  Visualize the top 36 eigenvectors found
subplot(1, 2, 2);
displayData(U(:, 1:36)');
title('top 36 eigenvectors');
axis square;
%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
%fprintf('\nDimension reduction for face dataset.\n\n');


K = 100;

Z = projectData(X_norm, U, K);
%displayData(Z(1:100,:));
%title('reduced data');
%axis square;
%fprintf('The projected data Z has a size of: ')
%fprintf('%d ', size(Z));

%fprintf('\n\nProgram paused. Press enter to continue.\n');
%pause;

%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

%fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

X_rec  = recoverData(Z, U, K);

figure
% Display normalized data
subplot(1, 3, 1);
displayData(dataMatrix(1:100,:));
title('Original gestures');
axis square;

subplot(1, 3, 2);
displayData(X_norm(1:100,:));
title('Normalised gestures');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 3, 3);
displayData(X_rec(1:100,:));
title('Recovered gestures');
axis square;
