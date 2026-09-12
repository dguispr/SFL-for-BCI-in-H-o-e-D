
tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = SubtractionLayer(1,"Name","SubtractionLayer");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = SubtractionLayer(2,"Name","addition");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = SubtractionLayer(1,"Name","SubtractionLayer");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = SubtractionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv");
lgraph = connectLayers(lgraph,"imageinput","conv_5");
lgraph = connectLayers(lgraph,"relu","conv_1");
lgraph = connectLayers(lgraph,"relu","conv_2");
lgraph = connectLayers(lgraph,"relu_1","multiplication/in1");
lgraph = connectLayers(lgraph,"relu_2","multiplication/in2");

lgraph = connectLayers(lgraph,"multiplication1", "SubtractionLayer");

lgraph = connectLayers(lgraph,"relu","addition/in1");
lgraph = connectLayers(lgraph,"SubtractionLayer","addition/in2");

lgraph = connectLayers(lgraph,"relu_5","conv_3");
lgraph = connectLayers(lgraph,"relu_5","conv_4");
lgraph = connectLayers(lgraph,"relu_3","multiplication_1/in1");
lgraph = connectLayers(lgraph,"relu_4","multiplication_1/in2");

lgraph = connectLayers(lgraph,"multiplication_1", "SubtractionLayer");

lgraph = connectLayers(lgraph,"relu_5","addition/in1");
lgraph = connectLayers(lgraph,"SubtractionLayer","addition/in2");
