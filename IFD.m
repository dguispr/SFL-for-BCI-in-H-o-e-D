
tempLayers = maxPooling2dLayer([4 4],"Name","maxpool","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([4 4],"Name","avgpool2d","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat");
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"maxpool","depthcat/in1");
lgraph = connectLayers(lgraph,"avgpool2d","depthcat/in2");
