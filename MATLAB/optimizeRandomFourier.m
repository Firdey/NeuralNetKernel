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
    
    %Nelder-Mead
%     [theta,T] = fminsearch(@objective_nested,theta_0);
    
    %Quasi-Newton
%     options = optimoptions('fminunc','Algorithm','quasi-newton');
%     [theta,T] = fminunc(@objective_nested,theta_0,options);

    %Trust-Region
%     options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');
%     [theta,T] = fminunc(@objective_nested,theta_0,options);
    
    %gradient descent
%     [theta,T] = gradientDescent(X,y,m,lambda,mu,activation,d_activation,theta_0,0);
    
    %stochastic gradient descent
    [theta,T] = gradientDescent(X,y,m,lambda,mu,activation,d_activation,theta_0,1);
  
    
    
    %SUBFUNCTION: GRADIENT DESCENT
    function [theta,T] = gradientDescent(X,y,m,lambda,mu,activation,d_activation,theta_0,isStochastic)
        %save inital value
        theta = theta_0;
        %declare a variable which stores the objective the step before
        T_old = 0;
        %total number of iterations
        n_iterations = 1E4;%10E6;
        %stop the loop if the change in objective between a step is less than tolerance
        if ~isStochastic
            tolerance = 10E-7;
        else
            tolerance = 10E-10;
        end
        %step size for gradient descent
        step_size = 0.1;
        n = numel(X(:,1));
        %for n_interations
        for i = 1:n_iterations
            %get the objective and the gradient
            if ~isStochastic
                [T,dT] = objective(X,y,m,lambda,mu,activation,d_activation,theta);
            else
                T = objective(X,y,m,lambda,mu,activation,d_activation,theta);
                stochastic_index = mod(i-1,n)+1;
                [~,dT] = objective(X(stochastic_index,:),y(stochastic_index),m,lambda,mu,activation,d_activation,theta);
            end
            %if the difference is less than tolerance and this is not the first iteration
            if ((abs(T-T_old)<tolerance) && (i~=1))
                %stop iterating
                break;
            end
            %if this is not the last iteration
            if (i ~= n_iterations)
                %gradient descent and save the objective
                theta = theta - step_size * dT;
                T_old = T;
            end
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