%Bharavi Misra
function [ outarray ] = apply_relu(inarray)
    outarray = max(inarray, 0);
end