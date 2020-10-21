classdef ReLU < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
    end
    
    methods
        function obj = ReLU()
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            output = max(0, input);
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            greaterThan0 = max(0, obj.outputCache);
            g = logical(greaterThan0) + 0;
            passBack = takeIn .* g;
        end
    end
end