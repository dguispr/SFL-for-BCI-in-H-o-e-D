
for sub = 1:9
        training = ['L:\BCI_IV_2a_exp\CWT_5_45Hz\sub_', num2str(sub), '\training'];
        path1 = imageDatastore(training, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        [training, valid] = splitEachLabel(path1,0.95, 0.05, 'randomized');
        
        path2 = ['L:\BCI_IV_2a_exp\CWT_5_45Hz\sub_', num2str(sub), '\testing'];
        testing = imageDatastore(path2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        
        %% Layers
        lgraph = layerGraph;
        tempLayers = imageInputLayer([1000 1 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 1],32,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([5 1],64,"Name","conv_1","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([2 1],"Name","avgpool2d","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 1],128,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([2 1],"Name","maxpool","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(64,"Name","fc")
    fullyConnectedLayer(4,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
        
        clear tempLayers;
        
        %% Connections
        lgraph = connectLayers(lgraph,"imageinput","conv");
        lgraph = connectLayers(lgraph,"imageinput","avgpool2d");
        lgraph = connectLayers(lgraph,"avgpool2d","depthcat/in3");
        lgraph = connectLayers(lgraph,"conv_1","conv_3");
        lgraph = connectLayers(lgraph,"conv_1","batchnorm_2");
        lgraph = connectLayers(lgraph,"relu_1","maxpool");
        lgraph = connectLayers(lgraph,"relu_1","multiplication/in2");
        lgraph = connectLayers(lgraph,"maxpool","depthcat/in2");
        lgraph = connectLayers(lgraph,"relu_3","multiplication/in1");
        lgraph = connectLayers(lgraph,"multiplication","depthcat/in1");
                      
                      
                     %% Training options
                        checkpointPath = ['L:\BCI_IV_2a_exp\CWT_5_45Hz\CNN_2D\chkPoints_',num2str(sub)];
                        options = trainingOptions('adam',...
                        'InitialLearnRate',0.0001, 'MiniBatchSize',64,...
                        'MaxEpochs', 200,...
                        'Shuffle','every-epoch', ...
                        'BatchNormalizationStatistics','moving',...
                        'Plots','training-progress',...
                        'CheckpointPath',checkpointPath,...
                        'ValidationData', valid, ...
                        'ValidationFrequency', 1, ...
                        'ValidationPatience',Inf, ...
                        'OutputNetwork','best-validation-loss');
                        
                    %% data augmentation
                        imageAugmenter = imageDataAugmenter( ...
                        'RandRotation',[-20, 20], ...
                        'RandScale',[0.5 1], ...
                        'RandXShear',[-5 5], ...
                        'RandXTranslation',[-30 30], ...
                        'RandYTranslation',[-10 10], ... 
                        'RandXReflection', 1);
                    
                    
                    %% results
                    [net,info] = trainNetwork(training,lgraph,options);
                    
                    %% save network/info
                    save(strcat('1DCNN', num2str(sub)), 'net')
                    save(strcat('1DCNN_Info', num2str(sub)), 'info')
        
                    [pred,probs] = classify(net,testing);
                    [c_matrix,Result]= confusionmat(testing.Labels,pred);
                    Accuracy = mean(pred == testing.Labels)*100;
                        CMat = c_matrix;
                        tp=CMat(1,1);
                        fp=CMat(2,1);
                        fn=CMat(1,2);
                        tn=CMat(2,2);
            
                        Precision = (tp./(tp+fp))*100;
                        Recall = (tp./(tp+fn))*100
                        specificity = (tn./(tn+fp))*100
                        F1_Score = (2*Precision*Recall)/(Precision+Recall)
end
