% Softmax layer that conform to Layer protocol
classdef SoftMaxLayer < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
        g
    end
    
    methods
        function obj = SoftMaxLayer()
        end
        
        % Forward propagation
        function output = forward(obj,input)
            obj.inputCache = input;
            expInput = exp(input);
            sumExpInput = sum(expInput, 1);
            output = expInput ./ sumExpInput;
            obj.outputCache = output;
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
%             outVal = obj.outputCache;
%             ai = repmat(outVal', obj.numOfInOut, 1);
%             aj = repmat(1-outVal, 1, obj.numOfInOut);
%             pbMatrix = ai .* aj - diag(ai);
%             passBack = pbMatrix * takeIn;
            batchSize = size(obj.outputCache, 2);
            passBackCache = [];
            for i = 1 : batchSize
                currentOutput = obj.outputCache(:, i);
                currentTakeIn = takeIn(:, i);
                aiajMatrix = currentOutput * currentOutput';
                currentg = - aiajMatrix + diag(currentOutput);
                currentPassBack = currentg * currentTakeIn;
                passBackCache = [passBackCache currentPassBack];
            end
            passBack = passBackCache;
        end
    end
end

