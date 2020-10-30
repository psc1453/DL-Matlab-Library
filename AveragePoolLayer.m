 classdef AveragePoolLayer < Layer
    properties
        inoutChannel
        kernelSize
        inputCache  % height        * width         * inoutChannel  * image
        outputCache % height        * width         * inoutChannel  * image
        kernels     % kernelSize    * kernelSize
    end
    
    methods
        function obj = AveragePoolLayer(inoutChannel, kernelSize)
            obj.inoutChannel = inoutChannel;
            obj.kernelSize = kernelSize;
            obj.kernels = ones(kernelSize, kernelSize) / (kernelSize ^ 2);
        end
        
        function output = forward(obj, input)
            obj.inputCache = input;
            sizeInput = size(input);
            sizeOut1 = sizeInput;
            sizeOut1(1) = sizeOut1(1) / obj.kernelSize;
            sizeOut2 = sizeOut1;
            sizeOut2(2) = sizeOut2(2) / obj.kernelSize;
            sumForRow = zeros(sizeOut1);
            for j = 1 : obj.kernelSize
                sumForRow = sumForRow + input(j : obj.kernelSize : end - obj.kernelSize + j, :, :, :);
            end
            sumForCol = zeros(sizeOut2);
            for i = 1 : obj.kernelSize
                sumForCol = sumForCol + sumForRow(:, i : obj.kernelSize : end - obj.kernelSize + i, :, :);
            end
            output = sumForCol / (obj.kernelSize ^ 2);
        end
        
        function passBack = backward(obj, takeIn, momentum, l2)
            iter = size(takeIn);
            if (length(iter) < 4)
                if (length(iter) < 3)
                    iter = [iter 1];
                end
                iter = [iter 1];
            end
            iter = iter(3 : 4);
            for i = 1 : iter(1)
                for j = 1 : iter(2)
                    passBack(:, :, i, j) = kron(takeIn(:, :, i, j), obj.kernels);
                end
            end
        end
    end
end

