classdef Loss
    
    methods (Static)
        function [loss, gradient] = SquareError(label, out)
            loss = (label - out) .^ 2;
            gradient = - 2 * (label - out);
        end
        
        function [loss, gradient] = CrossEntropy(label, out)
            loss = - label .* log(out) - (1 - label) .* log(1 - out);
            gradient = - label ./ out + (1 - label) ./ (1 - out);
        end
    end
end

