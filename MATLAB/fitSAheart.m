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
%get the dimensions
[n,p] = size(X);
%normalise X
X(:,2:end) = (X(:,2:end) - repmat(mean(X(:,2:end)),n,1)) ./ repmat(std(X(:,2:end)),n,1);

%tuning parameters
lambda = 0;
mu = 0;

%store T_optimal and error_optimal for every step of m
n_repeat = 4;
m_array = [2,4,8,16];
T_optimal_array = zeros(n_repeat,length(m_array));
error_optimal_array = zeros(n_repeat,length(m_array));
timing_array = zeros(n_repeat,length(m_array));

%for every m
for m_iteration = 1:length(m_array)
    m = m_array(m_iteration);
    
    %repeat n_repeat times
    parfor i = 1:n_repeat
        
        %start stopwatchq
        tic
        
        %initalize parameters
        theta = normrnd(0,1,2*m+m*p,1);

        %using optimization packages, find the theta which minimize the objective
        theta_optimal = optimizeRandomFourier(X,Y,m,lambda,mu,'sigmoid','d_sigmoid',theta);

        %get the optimal object and train error
        [T_optimal, error_optimal] = getT_Error(X,Y,m,lambda,mu,'sigmoid',theta_optimal);

        %store result in the array
        T_optimal_array(i,m_iteration) = T_optimal; %objective
        error_optimal_array(i,m_iteration) = error_optimal; %training error
        timing_array(i,m_iteration) = toc; %timing
    
    end
    
end

%plot the error_optimal for every m as a box plot
figure();
boxplot(error_optimal_array,'labels',m_array);
xlabel('Number of nodes in the hidden layer');
ylabel('Train error');

%plot the optimal objective for every m as a box plot
figure();
boxplot(T_optimal_array,'labels',m_array);
xlabel('Number of nodes in the hidden layer');
ylabel('Objective');

%plot the timing for every m as a box plot
figure();
boxplot(timing_array,'labels',m_array);
xlabel('Number of nodes in the hidden layer');
ylabel('Time took to optimize (s)');