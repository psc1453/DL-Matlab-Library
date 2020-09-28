classdef Sigmoid < handle
    properties
        inputCache % in * batch
        outputCache % out * batch
        numOfInOut
        g
    end
    
    methods
        function obj = Sigmoid(numOfInOut)
            obj.numOfInOut = numOfInOut;
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            output = 1 ./ (1 + exp(-input));
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum) % passBack: in * batch, takeIn: out * batch
            passBack = takeIn .* (1 ./ (1 + exp(-obj.inputCache))) .* (1 - 1 ./ (1 + exp(-obj.inputCache)));
            obj.g = passBack;
        end
        
        function step(obj, lr)
        end
    end
end

