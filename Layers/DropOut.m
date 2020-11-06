classdef DropOut < Layer
    properties
        inputCache % in * batch
        outputCache % out * batch
        numOfInOut
        ratio
        drop
        isTrain
    end
    
    methods
        function obj = DropOut(ratio)
            obj.ratio = ratio;
            obj.isTrain = 1;
        end
        
        function output = forward(obj,input)
            obj.inputCache = input;
            obj.numOfInOut = size(input, 1);
%             obj.drop = randsrc(obj.numOfInOut, 1, [1 0; 0.5 0.5]);
            if (obj.isTrain)
                obj.drop = (randi(1000, obj.numOfInOut, 1) > (1000 * obj.ratio));
                output = input .* obj.drop / (1 - obj.ratio);
            else
                output = input;
            end
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            passBack = takeIn .* obj.drop / (1 - obj.ratio);
        end
    end
end

