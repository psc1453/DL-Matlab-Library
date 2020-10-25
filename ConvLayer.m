 classdef ConvLayer < Layer
    properties
        inputChannel
        outputChannel
        kernelSize
        inputCache % height * width * inputChannel * image
        outputCache % height * width * outputChannel * image
        kernels % kernelSize * kernelSize * inputChannel * outputChannel
        gKernels % kernelSize * kernelSize * inputChannel * outputChannel
    end
    
    methods
        function obj = ConvLayer(inputChannel, outputChannel, kernelSize)
            obj.inputChannel = inputChannel;
            obj.outputChannel = outputChannel;
            obj.kernelSize = kernelSize;
            obj.kernels = randn(kernelSize, kernelSize, inputChannel, outputChannel)
            obj.gKernels = zeros(kernelSize, kernelSize, inputChannel, outputChannel)
        end
        
        function output = forward(obj, input)
            obj.inputCache = input;
            for i = 1 : obj.outputChannel
                output(:, :, i, :) = convn(input, obj.kernels(:, :, :, i), 'valid');
            end
            obj.outputCache = output;
        end
        
        function passBack = backward(obj, takeIn, momentum, l2)
            gForEveryOutChannel = [];
            for i = 1 : obj.outputChannel
                takeInTemp(:, :, 1, :) = takeIn(:, :, i, :);
                gForEveryOutChannel(:, :, :, i) = convn(obj.inputCache, takeInTemp, 'valid');
            end
            obj.gKernels = gForEveryOutChannel ./ size(takeIn, 4) + momentum * obj.gKernels;
            obj.kernels = (1 - l2) * obj.kernels - obj.gKernels;
            passBack = 0;
        end
    end
end

