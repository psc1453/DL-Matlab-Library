classdef model < handle
    properties
        layers
        numOfLayers
    end
    
    methods
        function obj = model(layers)
            obj.layers = layers;
            obj.numOfLayers = length(layers);
        end
        
        function output = forward(obj, input)
            data = input;
            for i = 1 : obj.numOfLayers
                data = obj.layers{i}.forward(data);
            end
            output = data;
        end
        
        function backward(obj, loss, lr, momentum, l2)
            passBack = loss * lr;
            for i = 1 : obj.numOfLayers
                passBack = obj.layers{obj.numOfLayers - i + 1}.backward(passBack, momentum, l2);
            end
        end
    end
end

