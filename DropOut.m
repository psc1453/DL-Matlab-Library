classdef DropOut < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
        numOfInOut
        drop
    end
    
    methods
        function obj = DropOut(numOfInOut)
            obj.numOfInOut = numOfInOut;
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
%             obj.drop = randsrc(obj.numOfInOut, 1, [1 0; 0.5 0.5]);
            obj.drop = (randi(1000, obj.numOfInOut, 1) > 500);
            output = input .* obj.drop;
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            passBack = takeIn .* obj.drop;
        end
    end
end

