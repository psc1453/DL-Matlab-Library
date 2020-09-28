% clear
% clc
% 
% dataSet = {[0 0 1 1;
%             0 1 0 1;
%             1 1 1 1], [0 1 1 0]};
% [data, label, numOfBatch] = generateBatch(dataSet{1}, dataSet{2}, 2);
% net = model({LinearLayer(3, 18), Sigmoid(18), ...
%              LinearLayer(18, 12), Sigmoid(12), ...
%              LinearLayer(12, 5), Sigmoid(5), ...
%              LinearLayer(5, 1), Sigmoid(1)});
% for i = 1 : 10000
%     for j = 1 : numOfBatch
%         out = net.forward(data{j});
%         loss = 2 * (label{j} - out);
%         net.backward(loss, 0.8);
%     end
% end
% result = net.forward(dataSet{1})






clear
clc

dataSet = {[0 0 1 1;
            0 1 0 1;
            1 1 1 1], [0 1 1 0;
                       1 1 0 1]};
[data, label, numOfBatch] = generateBatch(dataSet{1}, dataSet{2}, 2);

net = model({LinearLayer(3, 18), Sigmoid(18), ...
             LinearLayer(18, 12), Sigmoid(12), ...
             LinearLayer(12, 5), Sigmoid(5), ...
             LinearLayer(5, 2), Sigmoid(2)});
         
epoch = 1000;
printableLoss = [];

for i = 1 : epoch
    for j = 1 : numOfBatch
        out = net.forward(data{j});
        [loss, gradient] = Loss.CrossEntropy(label{j}, out);
        net.backward(gradient, 0.8, 0.9);
    end
    printableLoss = [printableLoss mean(loss)];
end
result = net.forward(dataSet{1})
% plot(1 : epoch, printableLoss, 'r-');
