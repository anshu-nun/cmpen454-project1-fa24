%Bharavi Misra
function [ outarray ] = apply_convolve(inarray, filterbank, biasvals)
    outarray = zeros(size(inarray,1), size(inarray,2), size(filterbank, 4));
    for i = 1:size(outarray,3)
        filter = filterbank(:,:,:,i);
        for j = 1:size(inarray,3)
            outarray(:,:,i) = outarray(:,:,i) + imfilter(inarray(:,:,j),filter(:,:,j),'conv', 'same');
        end
        outarray(:,:,i) = outarray(:,:,i) + biasvals(i);
    end
end