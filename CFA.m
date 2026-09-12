
tempLayers = imageInputLayer([227 227 3],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(1,"Name","SubtractionLayer");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv");
lgraph = connectLayers(lgraph,"imageinput","conv_1");
lgraph = connectLayers(lgraph,"imageinput","addition/in2");
lgraph = connectLayers(lgraph,"imageinput","multiplication_1/in2");
lgraph = connectLayers(lgraph,"relu","multiplication/in2");
lgraph = connectLayers(lgraph,"relu_1","multiplication/in1");
lgraph = connectLayers(lgraph,"multiplication", "SubtractionLayer");
lgraph = connectLayers(lgraph,"SubtractionLayer","addition/in1");
lgraph = connectLayers(lgraph,"addition","multiplication_1/in1");
