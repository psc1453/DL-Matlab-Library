classdef Sigmoid < handle
    properties
        g
        inputCache
        outputCache
        numOfInOut
    end
    
    methods
        function obj = Sigmoid(numOfInOut)
            obj.numOfInOut = numOfInOut;
        end

        function output = forward(obj, input)
            output = 1 ./ (1 + exp(-input));
            obj.outputCache = output;
            obj.inputCache = input;
        end
        
        function backward(obj)
            obj.g = 1 ./ (1 + exp(-obj.inputCache)) .* (1 - 1 ./ (1 + exp(-obj.inputCache)));
        end
        
        function passBack = step(obj, lr, batchSize)
            passBack = obj.g .* lr;
        end
        
        function zeroGrad(obj)
            obj.g = 0;
        end
    end
end

