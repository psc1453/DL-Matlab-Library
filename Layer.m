classdef Layer < handle
    methods (Abstract)
        forward(obj);
        backward(obj);
    end
end