classdef LeakyReLU < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
    end
    
    methods
        function obj = LeakyReLU()
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            g0 = (input >= 0) * 1;
            l0 = (input < 0) * 0.1;
            output = input .* (g0 + l0);
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            g0 = obj.outputCache >= 0;
            l0 = obj.outputCache < 0;
            g = g0 * 1 + l0 * 0.1;
            passBack = takeIn .* g;
        end
    end
end