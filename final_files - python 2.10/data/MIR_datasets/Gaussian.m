function data = Gaussian(N, opt, M)
% Generate MIR-Gaussian Data: y = f(x) + noise

% Inputs:
% N:   Number of bags
% opt: linear or nonlinear
% M:   Number of instances in each bag

% Output
% data: The synthetic data for MIR
%       data(:,1):          The bag information for each instance
%       data(:,2:end - 1):  Feature vector x
%       data(:,end):        Instance Label y (Instances in the same bag have the same label)

% reference: Z.Wang et.al 'Mixture Model for Multiple Instance Regression
%            and Applicatoins in Remote Sensing' 2011

x = rand(N,1);

if(strcmp(opt, 'linear'))
    y = x + randn(N,1)*0.05;
elseif(strcmp(opt, 'nonlinear'))
    y = x.^2 + randn(N,1)*0.05;
else
    error('invalid input for opt');    
end

data = [x y];
for i=1:size(data,1)
    a(i,1) = i;
end
data = [a data];

M = M - 1;
newdata = [];
for i=1:size(data,1)
    prime = data(i,2);
    nprime = prime + randn(M,1)*0.1;
    nprimeArray = [data(i,1)*ones(M,1) nprime data(i,3)*ones(M,1)];
    tmp = [data(i,:); nprimeArray];
    newdata = [newdata;tmp];
end

data = newdata;

% shuffle
[trash index] = sort(rand(size(data,1),1));
data = data(index,:);

return

