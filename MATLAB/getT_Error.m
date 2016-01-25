%FUNCTION GET OBJECTIVE T AND ERROR
function [T,error] = getT_Error(X,y,m,lambda,mu,activation,theta)
    
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

end

