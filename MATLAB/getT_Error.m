%FUNCTION GET OBJECTIVE T AND ERROR
function [T,train_error,test_error] = getT_Error(X_train,Y_train,X_test,Y_test,m,lambda,mu,activation,theta)
    
    %get the dimensions of X
    [n_train,p] = size(X_train);
    n_test = numel(X_test(:,1));
    
    %TRAINING ERROR
    
    %extract and reshape theta into the beta vector and Omega matrix
    beta = theta(1:2*m);
    Omega = reshape(theta((2*m+1):end),[m,p]);

    %regress using Omega as weights
    linear_combo = Omega*X_train'; % m x n matrix
    %non-linear transform using Fourier features
    Psy = [cos(linear_combo'),sin(linear_combo')]./sqrt(m); %n x 2m matrix
    %regress
    systematic = Psy*beta;
    %using the systematic component as the parameter for the activation
    yhat = feval(activation,systematic);

    %evaluate the objective and the train error
    T = sum((Y_train - yhat).^2)/n_train + lambda*sum(beta.^2) + mu*sum(sum(Omega.^2));
    train_error = sum(abs(Y_train-round(yhat)))/n_train;
    
    %TESTING ERROR
    
    %regress using Omega as weights
    linear_combo = Omega*X_test'; % m x n matrix
    %non-linear transform using Fourier features
    Psy = [cos(linear_combo'),sin(linear_combo')]./sqrt(m); %n x 2m matrix
    %regress
    systematic = Psy*beta;
    %using the systematic component as the parameter for the activation
    yhat = feval(activation,systematic);

    %evaluate the objective and the train error
    test_error = sum(abs(Y_test-round(yhat)))/n_test;

end

