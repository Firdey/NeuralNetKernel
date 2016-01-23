%FUNCTION: SIGMOID
function toReturn = sigmoid(x)
    toReturn = 1./(1+exp(-x));
end