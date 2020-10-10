classdef ReLU < handle
    properties
        inputCache % in * batch
        outputCache % out * batch
        numOfInOut
    end
    
    methods
        function obj = ReLU(numOfInOut)
            obj.numOfInOut = numOfInOut;
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            output = max(0.000001, input);
            output = min(0.999999, output);
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            greaterThan0 = max(0, obj.outputCache);
            g = logical(greaterThan0) + 0;
            passBack = takeIn .* g;
        end
    end
end