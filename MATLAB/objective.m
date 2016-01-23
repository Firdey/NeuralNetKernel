%FUNCTION: evaluation of the objective
function [T,dT,error] = objective(X,y,m,lambda,mu,activation,d_activation,theta)

    %get the dimensions of X
    [n,p] = size(X);

    %extract and reshape theta into the beta vector and Omega matrix
    beta = theta(1:2*m);
    Omega = reshape(theta((2*m+1):end),[m,p]);

    %regress using Omega as weights
    linear_combo = Omega*X'; % m x n matrix
    %non-linear transform using Fourier features
    Psy = [cos(linear_combo'),sin(linear_combo')]./sqrt(m); %n x 2m matrix
    %regress
    systematic = Psy*beta;
    %using the systematic component as the parameter for the activation
    yhat = feval(activation,systematic);

    %evaluate the objective and the train error
    T = sum((y - yhat).^2)/n + lambda*sum(beta.^2) + mu*sum(sum(Omega.^2));
    error = sum(abs(y-round(yhat)))/n;
    
    %evaluate the gradient
    dT = 0;
%     grad_S = feval(d_activation,systematic);
%     grad_beta_T = -2*Psy' * ((y-yhat).*grad_S) ./ n + 2*lambda.*beta;
%     D_Omega_T = zeros(m,p);
%     for i = 1:m
%         grad_omega_i_T = -(2/n) .* X' * ((y-yhat) .* grad_S .* (beta(m+i).*cos(Psy(:,i)) - beta(i).*sin(Psy(:,i))));
%         D_Omega_T(i,:) = grad_omega_i_T';
%     end
% 
%     dT = [grad_beta_T; reshape(D_Omega_T,[m*p,1])];
end