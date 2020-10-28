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
        function obj = ConvLayer(inputChannel, outputChannel, kernelSize)
            obj.inputChannel = inputChannel;
            obj.outputChannel = outputChannel;
            obj.kernelSize = kernelSize;
            obj.kernels(:, :, :, :, 1) = randn(kernelSize, kernelSize, inputChannel, outputChannel);
            obj.gKernels(:, :, :, :, 1) = zeros(kernelSize, kernelSize, inputChannel, outputChannel);
        end
        
        function output = forward(obj, input)
            obj.inputCache(:, :, :, 1, :) = input;
            for i = 1 : obj.outputChannel
                obj.outputCache(:, :, 1, i, :) = filtern(obj.inputCache(:, :, :, 1, :), obj.kernels(:, :, :, i, 1), 'valid');
            end
            outputSize = size(obj.outputCache);
            outputSize(3) = [];
            output = reshape(obj.outputCache, outputSize);
        end
        
        function passBack = backward(obj, takeIn, momentum, l2)
            delta(:, :, 1, :, :) = takeIn;
            passBackCacheSize = size(obj.inputCache);
            passBackCacheSize(4) = obj.outputChannel;
            passBackCache = zeros(passBackCacheSize);
            for i = 1 : obj.outputChannel
                passBackCache(:, :, :, i, :) = convn(delta(:, :, 1, i, :), obj.kernels(:, :, :, i, 1), 'full');
            end
            passBack = sum(passBackCache, 4);
            passBackSize = size(passBack);
            passBackSize(4) = [];
            passBack = reshape(passBack, passBackSize);
            gForEveryOutChannel = [];
            for i = 1 : obj.outputChannel
                gForEveryOutChannel(:, :, :, i, 1) = filtern(obj.inputCache(:, :, :, 1, :), delta(:, :, 1, i, :), 'valid');
            end
            obj.gKernels = gForEveryOutChannel ./ size(delta, 5) + momentum * obj.gKernels;
            obj.kernels = (1 - l2) * obj.kernels - obj.gKernels;  
        end
    end
end

