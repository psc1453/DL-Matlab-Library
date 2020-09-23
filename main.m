clear
clc

dataSet = {[0 0 1 1;
            0 1 0 1;
            1 1 1 1], [0 1 1 0]};
[data, label, numOfBatch] = generateBatch(dataSet{1}, dataSet{2}, 1);
net = model({LinearLayer(3, 4), Sigmoid(4), ...
             LinearLayer(4, 1), Sigmoid(1)});
for i = 1 : 10000
    for j = 1 : numOfBatch
        out = net.forward(data{j});
        loss = 2 * (label{j} - out);
        net.backward(loss, 0.8);
    end
end
result = net.forward(dataSet{1})