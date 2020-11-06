classdef Sigmoid < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
    end
    
    methods
        function obj = Sigmoid()
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            output = 1 ./ (1 + exp(-input));
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            passBack = takeIn .* (1 ./ (1 + exp(-obj.inputCache))) .* (1 - 1 ./ (1 + exp(-obj.inputCache)));
        end
    end
end

