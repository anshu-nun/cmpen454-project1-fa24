%June Kim
function [ outarray ] = apply_maxpool(inarray)
    [N, M, D] = size(inarray);
    outarray = zeros(N/2, M/2, D);
    for n = 1:N/2
        for m = 1: M/2
            for d = 1:D
                patch = inarray(2*n-1:2*n, 2*m-1:2*m, d);
                outarray(n, m, d) = max(patch(:));
            end
        end
    end
end