% Support batch
classdef ReLU < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
        k
    end
    
    % Example: For k is 1, using ReLU(1)
    methods
        function obj = ReLU(k)
            obj.k = k;
        end
        
        % Forward propagation
        function output = forward(obj,input)
            obj.inputCache = input;
            output = max(0, input * obj.k);
            obj.outputCache = output;
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            greaterThan0 = max(0, obj.outputCache);
            g = logical(greaterThan0) * obj.k;
            passBack = takeIn .* g;
        end
    end
end