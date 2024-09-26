%Bharavi Misra
function [ outarray ] = apply_imnormalize(inarray)
    outarray = double(inarray)/255.0 - 0.5;
end