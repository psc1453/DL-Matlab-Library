function [outData, outLabel, numOfBatch] = generateBatch(data, label, batchSize)
    dataSetCount = size(data, 2);
    numOfBatch = ceil(dataSetCount / batchSize);    
    miniBatchData = {};
    miniBatchLabel = {};
    for i = 1 : numOfBatch
        (i - 1) * batchSize : i * batchSize - 1;
        currentSelect = mod((i - 1) * batchSize : i * batchSize - 1, dataSetCount) + 1;
        miniBatchData = [miniBatchData data(:, currentSelect)];
        miniBatchLabel = [miniBatchLabel label(:, currentSelect)];
    end
    outData = miniBatchData;
    outLabel = miniBatchLabel;
end