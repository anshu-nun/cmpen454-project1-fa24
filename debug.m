%Bharavi Misra

%loading this file defines imrgb and layerResults
load 'debuggingTest.mat'

%loading this file defines filterbanks and biasvectors
load 'CNNparameters.mat'

%loading this file defines imageset, trueclass, and classlabels
load 'cifar10testdata.mat'

%sample code to verify which layers have filters and biases
for d = 1:length(layertypes)
    fprintf('layer %d is of type %s\n',d,layertypes{d});
    filterbank = filterbanks{d};
    if not(isempty(filterbank))
        fprintf('filterbank size %d x %d x %d x %d\n', ...
            size(filterbank,1),size(filterbank,2), ...
            size(filterbank,3),size(filterbank,4));
        biasvec = biasvectors{d};
        fprintf('number of bias vectors is %d\n',length(biasvec));
    end
end

computed = 0;

for d=1:length(layertypes)
    layer = layertypes{d};
    if strcmp(layer,'imnormalize')
        computed = apply_imnormalize(imrgb);
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d, totalDiff);
    end
    if strcmp(layer,'convolve')
        computed = apply_convolve(computed,filterbanks{d},biasvectors{d});
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d,totalDiff);
    end
    if strcmp(layer,'relu')
        computed = apply_relu(computed);
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d,totalDiff);
    end
    if strcmp(layer,'maxpool')
        computed = apply_maxpool(computed);
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d,totalDiff);
    end
    if strcmp(layer,'fullconnect')
        computed = apply_fullconnect(computed,filterbanks{d},biasvectors{d});
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d,totalDiff);
    end
    if strcmp(layer,'softmax')
        computed = apply_softmax(computed);
        expected = layerResults{d};
        diff = (computed-expected).^2;
        totalDiff = sum(diff(:));
        fprintf('difference between expected and observed values in layer %d is %0.2f\n', ...
                d,totalDiff);
    end

end

%sample code to show image and access expected results
%figure; imagesc(imrgb); truesize(gcf,[64 64]);
for d = 1:length(layerResults)
    result = layerResults{d};
    fprintf('layer %d output is size %d x %d x %d\n', ...
        d,size(result,1),size(result,2),size(result,3));
    fprintf('layer %d output is size %d x %d x %d\n', ...
        d,size(result,1),size(result,2),size(result,3));
end
%find expected most probable class
expected_classprobvec = squeeze(layerResults{end});
[expected_maxprob,expected_maxclass] = max(expected_classprobvec);
%note, classlabels is defined in 'cifar10testdata.mat'
fprintf('expected estimated class is %s with probability %.4f\n',...
    classlabels{expected_maxclass},expected_maxprob);
%find computed most probable class
computed_classprobvec = squeeze(computed);
[computed_maxprob,computed_maxclass] = max(computed_classprobvec);
%note, classlabels is defined in 'cifar10testdata.mat'
fprintf('expected computed class is %s with probability %.4f\n',...
    classlabels{computed_maxclass},computed_maxprob);
