% Andrei Dawinan - abd5635@psu.edu

function [ outarray ] = apply_fullconnect(inarray, filterbank, biasvals)
    outarray = zeros(1, 1, size(filterbank, 4));
    % filters: index from 1 to D2
    for filters = 1:size(filterbank, 4)
        filter = filterbank(:,:,:,filters);
        % element-wise multiplication per filter + bias
        outarray(:,:,filters) = sum(inarray .* filter, "all") + biasvals(filters);
    end
end