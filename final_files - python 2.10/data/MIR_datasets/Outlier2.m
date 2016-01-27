function data = Outlier2(N, opt, M)
% Generate MIR-Outlier2 Data

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

y(1:20)=y(1:20)+randn(20,1)*0.25;
 
bias = rand(N,1)*0.5-0.25;
t = rand(N,1)*0.5;

data = [x y];
for i=1:size(data,1)
    a(i,1) = i;
end
data = [a data];

newdata = [];
for i=1:size(data,1)
    prime = data(i,2);

    nprime1 = prime + randn(round(M*(1-t(i))),1)*0.1;
    nprime2 = prime + randn(M-round(M*(1-t(i))),1)*0.5+ bias(i);
    nprime = [nprime1;nprime2];

    nprimeArray = [data(i,1)*ones(M,1) nprime data(i,3)*ones(M,1)];
    tmp = [nprimeArray];
    newdata = [newdata;tmp];
end

data = newdata;

% shuffle
[trash index] = sort(rand(size(data,1),1));
data = data(index,:);

return
% index = unique(data(:,1));
% 
% for i=1:length(index)
%     id = index(i);
%     tmp = data(find(data(:,1)==id),:);
%  
%     plot(tmp(:,2),tmp(:,3),'.');
%     hold on
% end

% % no attribute noise
% res = ols(y,[ones(N,1) x])
% 
% xtest = (0:0.01:1)';
% plot(xtest,res.beta(1) + res.beta(2)*xtest, 'k')
% hold
% plot(x,y,'.')
% 
% % with attribute noise
% xnoisy =  x + randn(N,1)*1.0;
% res = ols(y,[ones(N,1) xnoisy])
% 
% plot(xtest,res.beta(1) + res.beta(2)*xtest, 'r')
