
classdef SP_CEL < nnet.layer.ClassificationLayer
    properties
        % No additional properties needed
    end
    
    methods
        function layer = SP_CEL(name)
            % Constructor to set the layer name
            layer.Name = name;
            layer.Description = 'Binary cross-entropy loss for 2-class classification';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Compute binary cross-entropy loss
            % Y: predicted probabilities (output of sigmoid)
            % T: true labels (one-hot encoded)

            % Ensure numerical stability by adding a small epsilon
            epsilon = 1e-12;
            Y = min(max(Y, epsilon), 1 - epsilon); 

            rho = Y/(1-Y);
            % Compute binary cross-entropy loss
            loss = -mean(T .* log(Y)^(1/rho) + (1 - T) .* log(1 - Y)^rho, 'all');
        end
    end
end
