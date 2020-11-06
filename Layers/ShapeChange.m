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
        
        function output = forward(obj, input)
            elements = numel(input);
            lastDimension = elements / prod(obj.outputShape);
            shape = [obj.outputShape lastDimension];
            output = reshape(input, shape);
        end
        
        function passBack = backward(obj, takeIn, momentum, l2) % passBack: in * batch, takeIn: out * batch
            elements = numel(takeIn);
            lastDimension = elements / prod(obj.inputShape);
            shape = [obj.inputShape lastDimension];
            passBack = reshape(takeIn, shape);
        end
    end
end

