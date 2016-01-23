%FUNCTION: DIFFERENTIAL OF SIGMOID
function toReturn = d_sigmoid(x)
    toReturn = exp(-x)./((1+exp(-x)).^2);
end