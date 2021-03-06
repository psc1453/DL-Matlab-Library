% Layer for extract feature
% Support batch
classdef ConvLayer < Layer
    properties
        inputChannel
        outputChannel
        kernelSize
        inputCache  % height        * width         * inputChannel  * 1                 * image
        outputCache % height        * width         * 1             * outputChannel     * image
        kernels     % kernelSize    * kernelSize    * inputChannel  * outputChannel     * 1
        gKernels    % kernelSize    * kernelSize    * inputChannel  * outputChannel     * 1
    end
    
    methods
        % Example: For 3 input channel, 10 output channel, 5x5 kernel size, using ConvLayer(3, 10, 5)
        function obj = ConvLayer(inputChannel, outputChannel, kernelSize)
            obj.inputChannel = inputChannel;
            obj.outputChannel = outputChannel;
            obj.kernelSize = kernelSize;
            obj.kernels(:, :, :, :, 1) = (randn(kernelSize, kernelSize, inputChannel, outputChannel)) / (inputChannel * kernelSize ^ 2);
            obj.gKernels(:, :, :, :, 1) = zeros(kernelSize, kernelSize, inputChannel, outputChannel);
        end
        
        % Forward propagation
        function output = forward(obj, input)
            obj.inputCache = [];
            obj.outputCache = [];
            obj.inputCache(:, :, :, 1, :) = input;
            for i = 1 : obj.outputChannel
                obj.outputCache(:, :, 1, i, :) = filtern(obj.inputCache(:, :, :, 1, :), obj.kernels(:, :, :, i, 1), 'valid');
            end
            outputSize = size(obj.outputCache);
            % Delete the 3rd dimension which is used for allignment
            if (length(outputSize) > 2)
                outputSize(3) = [];
            end
            output = reshape(obj.outputCache, outputSize);
        end
        
        % Backward propagation
        function passBack = backward(obj, takeIn, momentum, l2)
            delta(:, :, 1, :, :) = takeIn;
%             passBackCacheSize = size(obj.inputCache);
%             passBackCacheSize(4) = obj.outputChannel;
%             passBackCache = zeros(passBackCacheSize);
            passBackCache = 0;
            for i = 1 : obj.outputChannel
                passBackCache = passBackCache + convn(delta(:, :, 1, i, :), obj.kernels(:, :, :, i, 1), 'full');
            end
            passBack = sum(passBackCache, 4);
            passBackSize = size(passBack);
            if (length(passBackSize) > 3)
                passBackSize(4) = [];
            end
            passBack = reshape(passBack, passBackSize);
            gForEveryOutChannel = [];
            for i = 1 : obj.outputChannel
                gForEveryOutChannel(:, :, :, i, 1) = filtern(obj.inputCache(:, :, :, 1, :), delta(:, :, 1, i, :), 'valid');
            end
            obj.gKernels = gForEveryOutChannel ./ size(delta, 5) + momentum * obj.gKernels;
            obj.kernels = obj.kernels - obj.gKernels;  
        end
    end
end

