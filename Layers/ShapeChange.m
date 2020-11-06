% A reshape layer that can auto dealing with batch size
% Also, conform to Layer protocol that can be used to auto propagation
classdef ShapeChange < Layer
    properties
        inputShape
        outputShape
    end
    
    methods
        
        function obj = ShapeChange(inputShape, outputShape)
            obj.inputShape = inputShape;
            obj.outputShape = outputShape;
        end
        
        % Forward propagation
        function output = forward(obj, input)
            elements = numel(input);
            lastDimension = elements / prod(obj.outputShape);
            shape = [obj.outputShape lastDimension];
            output = reshape(input, shape);
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            elements = numel(takeIn);
            lastDimension = elements / prod(obj.inputShape);
            shape = [obj.inputShape lastDimension];
            passBack = reshape(takeIn, shape);
        end
    end
end

