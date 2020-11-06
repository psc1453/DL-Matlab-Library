% A model for users to construct and use neural networks.
classdef model < handle
    properties
        layers % Store layer structure of the model
        numOfLayers % Irrelevant to users, inner property
    end
    
    methods
        % For constructing model, accept layers in cell
        % Example: net = model({FullConnect(3, 1), ReLU(1)})
        function obj = model(layers)
            obj.layers = layers;
            obj.numOfLayers = length(layers);
        end
        
        % Auto forward calculation throuth every layer
        function output = forward(obj, input)
            data = input;
            for i = 1 : obj.numOfLayers
                data = obj.layers{i}.forward(data);
            end
            output = data;
        end
        
        % Auto backward propagation through every layer
        % Also Capable for adjust learing rate, momentum and L2-Regularization
        function backward(obj, loss, lr, momentum, l2)
            passBack = loss * lr;
            for i = 1 : obj.numOfLayers
                passBack = obj.layers{obj.numOfLayers - i + 1}.backward(passBack, momentum, l2);
            end
        end
        
        % Set the model to train mode
        function trainMode(obj)
            for i = 1 : length(obj.layers)
                if (isa(obj.layers{i}, 'DropOut'))
                    obj.layers{i}.isTrain = 1;
                end
            end
        end
        
        % Set the model to test mode
        function TestMode(obj)
            for i = 1 : length(obj.layers)
                if (isa(obj.layers{i}, 'DropOut'))
                    obj.layers{i}.isTrain = 0;
                end
            end
        end
    end
end

