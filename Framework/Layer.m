% Defined protocol for layers in this framework.
% Every layer should at least be able to propagate data bidirectionally.
classdef Layer < handle
    methods (Abstract)
        forward(obj); % A layer should be able to calculate an output
        backward(obj); % A layer should be able to propagate error backwards and if necessarry, adjust parameters.
    end
end