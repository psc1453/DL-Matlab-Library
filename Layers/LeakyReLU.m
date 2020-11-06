% Support batch
classdef LeakyReLU < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
        kn
        kp
    end
    
    % Example: For negtive input k is 0.1 for positive is 1, using LeakyReLU(0.1, 1)
    methods
        function obj = LeakyReLU(kn, kp)
            obj.kn = kn;
            obj.kp = kp;
        end
        
        % Forward propagation
        function output = forward(obj,input)
            obj.inputCache = input;
            g0 = (input >= 0) * obj.kp;
            l0 = (input < 0) * obj.kn;
            output = input .* (g0 + l0);
            obj.outputCache = output;
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            g0 = obj.outputCache >= 0;
            l0 = obj.outputCache < 0;
            g = g0 * obj.kp + l0 * obj.kn;
            passBack = takeIn .* g;
        end
    end
end