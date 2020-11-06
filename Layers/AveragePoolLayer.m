% Layer for avarage pooling
% Support batch
classdef AveragePoolLayer < Layer
    properties
        inoutChannel
        kernelSize
        inputCache  % height        * width         * inoutChannel  * image
        outputCache % height        * width         * inoutChannel  * image
        kernels     % kernelSize    * kernelSize
    end
    
    methods
        % Example: For 3 channel, 2x2 average pooling, using AveragePoolLayer(3, 2)
        function obj = AveragePoolLayer(inoutChannel, kernelSize)
            obj.inoutChannel = inoutChannel;
            obj.kernelSize = kernelSize;
            obj.kernels = ones(kernelSize, kernelSize) / (kernelSize ^ 2);
        end
        
        % Forward propagation
        % Algorithm: take 4x4 image using 2x2 pooling, 
        % 1. Sum row 1,3 with row 2,4, the image become 2x4
        % 2. Sum col 1,3 with col 2,4, the image become 2x2
        % 3. Divide the result by 2x2
        function output = forward(obj, input)
            obj.inputCache = input;
            sizeInput = size(input);
            sizeOut1 = sizeInput;
            sizeOut1(1) = sizeOut1(1) / obj.kernelSize; % Size after first step
            sizeOut2 = sizeOut1;
            sizeOut2(2) = sizeOut2(2) / obj.kernelSize; % Size after second step
            % First step
            sumForRow = zeros(sizeOut1);
            for j = 1 : obj.kernelSize
                sumForRow = sumForRow + input(j : obj.kernelSize : end - obj.kernelSize + j, :, :, :);
            end
            % Second step
            sumForCol = zeros(sizeOut2);
            for i = 1 : obj.kernelSize
                sumForCol = sumForCol + sumForRow(:, i : obj.kernelSize : end - obj.kernelSize + i, :, :);
            end
            % Third step
            output = sumForCol / (obj.kernelSize ^ 2);
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2)
            iter = size(takeIn);
            % Since matlab will represent dimension like 4x3x1x1 as 4x3
            % Cannot refer to the unexist 3rd and 4th element of the size
            % Full fill the ignored dimensions with 1
            if (length(iter) < 4)
                if (length(iter) < 3)
                    iter = [iter 1];
                end
                iter = [iter 1];
            end
            iter = iter(3 : 4);
            % For every channel in one image
            for i = 1 : iter(1)
                % For every image in one batch
                for j = 1 : iter(2)
                    passBack(:, :, i, j) = kron(takeIn(:, :, i, j), obj.kernels);
                end
            end
        end
    end
end

