 classdef LinearLayer < handle
    properties
        numOfInput
        numOfOutput
        inputCache % in * batch
        outputCache % out * batch
        gW % out * in
        gB % out * 1
        W % out * in
        B % out * 1
    end
    
    methods
        function obj = LinearLayer(numOfInput, numOfOutput)
            obj.numOfInput = numOfInput;
            obj.numOfOutput = numOfOutput;  
            obj.gW = zeros(numOfOutput, numOfInput);
            obj.gB = zeros(numOfOutput, 1);
            obj.W = 2 * rand(numOfOutput, numOfInput) - 1;
            obj.B = zeros(numOfOutput, 1);
        end
        
        function output = forward(obj, input)
            obj.inputCache = input;
            output = obj.W * input;
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum) % passBack: in * batch, takeIn: out * batch
            delta = takeIn;
            passBack = obj.W' * delta;
            obj.gW = (obj.inputCache * delta')' / size(obj.inputCache, 2) + momentum * obj.gW;
            obj.gB = mean(takeIn, 2) + momentum * obj.gB;
            obj.W = obj.W - obj.gW;
            obj.B = obj.B - obj.gB;
        end
    end
end

