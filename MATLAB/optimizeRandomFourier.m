%FUNCTION: OPTIMIZE THE OBJECTIVE FOR RANDOM FOURIER REPRESENTATION
%PARAMETERS:
    %X: design matrix
    %y: response matrix
    %lambda: tuning parameter for beta
    %mu: tuning parameter for weights
    %activation: activation function to predict the response
    %theta_0: initial value
function [theta,T] = optimizeRandomFourier(X,y,m,lambda,mu,activation,d_activation,theta_0)

    %using the nested function objective, minimize it and return the
    %optimizing parameter theta, and the objective T
    %options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');
    %options = optimoptions('fminunc','Algorithm','quasi-newton');
    %[theta,T] = fminunc(@objective_nested,theta_0,options);
    %[theta,T] = fminsearch(@objective_nested,theta_0);
    
    theta = theta_0;
    T_old = 0;
    n_interations = 10E6;
    for i = 1:n_interations
        [T,dT] = objective(X,y,m,lambda,mu,activation,d_activation,theta);
        if ((abs(T-T_old)<10E-7) && (i~=1))
            disp('break!');
            break;
        end
        if (i ~= n_interations)
            theta = theta - 0.1 * dT;
            T_old = T;
        end
    end
    
    %evaluation of the objective
    function [T,dT] = objective_nested(theta)
        if nargout == 1
            T = feval('objective',X,y,m,lambda,mu,activation,d_activation,theta);
        else
            [T,dT] = feval('objective',X,y,m,lambda,mu,activation,d_activation,theta);
        end
    end

end