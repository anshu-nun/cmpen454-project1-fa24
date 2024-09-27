%Anshu Nunemunthala - June Kim

%defines filterbanks and biasvectors
load './CNNparameters.mat'

%efines imageset, trueclass, and classlabels
load './cifar10testdata.mat'

%initialze confusion matrix C
C = zeros(10,10);

%iterate through all images in cifar10testdata
for i=1:size(imageset,4)
    %we iterate through all 18 layers of the network and keep track of the output array at each one
    layerResults = cell(1,length(layertypes));
    for j=1:length(layertypes)
        layer = layertypes{j};
        if strcmp(layer,'imnormalize')
            layerResults{j} = apply_imnormalize(imageset(:,:,:,i));
        elseif strcmp(layer,'convolve')
            layerResults{j} = apply_convolve(layerResults{j-1},filterbanks{j},biasvectors{j});
        elseif strcmp(layer,'relu')
            layerResults{j} = apply_relu(layerResults{j-1});
        elseif strcmp(layer,'maxpool')
            layerResults{j} = apply_maxpool(layerResults{j-1});
        elseif strcmp(layer,'fullconnect')
            layerResults{j} = apply_fullconnect(layerResults{j-1},filterbanks{j},biasvectors{j});
        elseif strcmp(layer,'softmax')
            layerResults{j} = apply_softmax(layerResults{j-1});
        end
    end
    %find computed most probable class and true most probable class and update the confusion matrix
    classprobvec = squeeze(layerResults{end});
    [maxprob, maxclass] = max(classprobvec);
    C(trueclass(i),maxclass) = C(trueclass(i),maxclass) + 1;
end

%compute classifier accuracy
accuracy = trace(C)/sum(C(:));
fprintf("The matrix accuracy is %0.2f\n", accuracy);
