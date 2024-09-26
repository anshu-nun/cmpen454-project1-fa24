%Bharavi Misra
function [outarray] = apply_softmax(inarray)
    outarray = zeros(1, 1, size(inarray,3));
    alpha = max(inarray);
    denominator = sum(exp(inarray-alpha));
    for i=1:size(outarray,3)
        outarray(:,:,i) = exp(inarray(:,:,i)-alpha)/denominator;
    end
end