classdef LinearLayer < handle
    properties
        W
        B
        gW
        gB
        inputCache
        outputCache
        numOfIn
        numOfOut
    end
    
    methods
        function obj = LinearLayer(numOfIn,numOfOut)
            obj.W = 2 * rand(numOfOut, numOfIn) - 1;
            obj.B = zeros(numOfOut, 1); 
            obj.numOfIn = numOfIn;
            obj.numOfOut = numOfOut;
        end

        function output = forward(obj, input)
            output = obj.W * input + obj.B;
            obj.outputCache = output;
            obj.inputCache = input;
        end
        
        function backward(obj)
            inputT = obj.inputCache';
            obj.gW = repmat(inputT, [obj.numOfOut, 1]);
            obj.gB = ones(obj.numOfOut, 1);
        end
        
        function passBack = step(obj, lr)
            passBack = (lr .* obj.W)' * obj.outputCache;
            obj.W = obj.W + lr .* obj.gW;
            obj.B = obj.B + lr .* obj.gB;
        end
        
        function zeroGrad(obj)
            obj.gW = 0;
            obj.gB = 0;
        end
    end
end

