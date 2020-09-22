classdef model < handle
    
    properties
        layers
        numOfLayer
    end
    
    methods
        function obj = model(layers)
            obj.layers = layers;
            obj.numOfLayer = length(obj.layers);
        end
        
        function data = forward(obj, data)
            for i = 1 : obj.numOfLayer
                data = obj.layers{i}.forward(data);
            end           
        end
        
        function backward(obj)
            for i = 1 : obj.numOfLayer
                obj.layers{obj.numOfLayer - i + 1}.backward();
            end
        end
        
        function step(obj, lr)
            for i = 1 : obj.numOfLayer
                lr = obj.layers{obj.numOfLayer - i + 1}.step(lr);
            end
        end
        
        function zeroGrad(obj)
            for i = 1 : obj.numOfLayer
                obj.layers{obj.numOfLayer - i + 1}.zeroGrad();
            end
        end
    end
end

