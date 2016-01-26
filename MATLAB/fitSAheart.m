clc;
clear all;
close all;

%load the data from the csv file
data = csvread('SAheart.csv',1,0);
%get the design matrix
X = data(:,1:(end-1));
%add constant term
X(:,1) = 1;
%get the response vector
Y = data(:,end);
%get the dimensions and size of training set
[n,p] = size(X);
n_train = round(n/2);
%normalise X
X(:,2:end) = (X(:,2:end) - repmat(mean(X(:,2:end)),n,1)) ./ repmat(std(X(:,2:end)),n,1);

%tuning parameters
lambda = 0;
mu = 0;

%store T_optimal and error_optimal for every step of m
n_repeat = 48;
m_array = [2,4,8,16,32,64];
trainingError_array = zeros(n_repeat,length(m_array));
testError_array = zeros(n_repeat,length(m_array));
timing_array = zeros(n_repeat,length(m_array));

%split into test and train set
% rng(1070);
% index = randperm(n);
% rng('shuffle');
% X_train = X(index(1:n_train),:);
% X_test = X(index((n_train+1):end),:);
% Y_train = Y(index(1:n_train));
% Y_test = Y(index((n_train+1):end));

%for every m
for m_iteration = 1:length(m_array)
    m = m_array(m_iteration);
    
    %repeat n_repeat times
    parfor i = 1:n_repeat
        
        %split data
        index = randperm(n);
        X_train = X(index(1:n_train),:);
        X_test = X(index((n_train+1):end),:);
        Y_train = Y(index(1:n_train));
        Y_test = Y(index((n_train+1):end));
        
        %initalize parameters
        theta = normrnd(0,1,2*m+m*p,1);

        %using optimization packages, find the theta which minimize the objective
        %start stopwatch
        tic
        theta_optimal = optimizeRandomFourier(X_train,Y_train,m,lambda,mu,'sigmoid','d_sigmoid',theta);
        timing_array(i,m_iteration) = toc; %stop stop watch and store time

        %get the optimal object and train error
        training_error = objective(X_train,Y_train,m,lambda,mu,'sigmoid','d_sigmoid',theta_optimal);
        test_error = objective(X_test,Y_test,m,lambda,mu,'sigmoid','d_sigmoid',theta_optimal);

        %store result in the array
        trainingError_array(i,m_iteration) = training_error; %training error
        testError_array(i,m_iteration) = test_error; %testing error
    
    end
    disp(m);
end

figure('Position', [800, 600, 700, 420]);

%plot the training error for every m as a box plot
subplot(1,3,1);
boxplot(trainingError_array,'labels',m_array);
xlabel('m');
ylabel('Training error');

%plot the test error for every m as a box plot
subplot(1,3,2);
boxplot(testError_array,'labels',m_array);
xlabel('m');
ylabel('Test error');

%plot the timing for every m as a box plot
subplot(1,3,3);
boxplot(timing_array,'labels',m_array);
xlabel('m');
ylabel('Time took to optimize (s)');