% Support batch
classdef FullConnectLayer < Layer
    properties
        numOfInput
        numOfOutput
        inputCache % in * batch
        outputCache % out * batch
        gW % out * in
%         gB % out * 1
        W % out * in
%         B % out * 1
    end
    
    
    methods
        % Example: For 3 input, 10 output, using FullConnectLayer(3, 10)
        function obj = FullConnectLayer(numOfInput, numOfOutput)
            obj.numOfInput = numOfInput;
            obj.numOfOutput = numOfOutput;  
            obj.gW = zeros(numOfOutput, numOfInput);
%             obj.gB = zeros(numOfOutput, 1);
            obj.W = (2 * rand(numOfOutput, numOfInput) - 1) / numOfInput;
%             obj.B = zeros(numOfOutput, 1);
        end
        
        % Forward Propagation
        function output = forward(obj, input)
            obj.inputCache = input;
            output = obj.W * input;
            obj.outputCache = output;
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            delta = takeIn;
            passBack = obj.W' * delta;
            obj.gW = (obj.inputCache * delta')' / size(obj.inputCache, 2) + momentum * obj.gW;
%             obj.gB = mean(takeIn, 2) + momentum * obj.gB;
            obj.W = (1 - l2) * obj.W - obj.gW;
%             obj.B = obj.B - obj.gB;
        end
    end
end

