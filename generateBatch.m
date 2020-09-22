function [outData, outLabel, elementNumOfBatch] = generateBatch(data, label, batchSize)
    elementNumOfBatch = ceil(size(data, 1) / batchSize);
    dataSetCount = size(data, 1);
    miniBatchData = {};
    miniBatchLabel = {};
    for i = 1 : elementNumOfBatch
        currentSelect = mod((i - 1) * batchSize : i * batchSize - 1, dataSetCount) + 1;
        pickedData = data(currentSelect, :)';
        oneBatchData = reshape(pickedData, [length(pickedData(:)) / elementNumOfBatch, 1, elementNumOfBatch]);
        pickedLabel = label(currentSelect, :)';
        oneBatchLabel = reshape(pickedLabel, [length(pickedLabel(:)) / elementNumOfBatch, 1, elementNumOfBatch]);
        miniBatchData = [miniBatchData oneBatchData];
        miniBatchLabel = [miniBatchLabel oneBatchLabel];
%         currentSelect = mod((i - 1) * batchSize : i * batchSize - 1, dataSetCount) + 1;
%         data(currentSelect, :)
%         label(currentSelect, :)
%         miniBatchData = [miniBatchData data(currentSelect, :)];
%         miniBatchLabel = [miniBatchLabel label(currentSelect, :)];
    end
    outData = miniBatchData;
    outLabel = miniBatchLabel;
end
