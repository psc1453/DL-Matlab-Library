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

% dataSet = {[0 0 1 1;
%             0 1 0 1;
%             1 1 1 1], [0 1 1 0;
%                        1 0 0 1]};

data = [];
data(:, 1) = reshape([0 1 1 0 0;
                      0 0 1 0 0;
                      0 0 1 0 0;
                      0 0 1 0 0;
                      0 1 1 1 0], 25, 1);
                  
data(:, 2) = reshape([1 1 1 1 0;
                      0 0 0 0 1;
                      0 1 1 1 0;
                      1 0 0 0 0;
                      1 1 1 1 1], 25, 1);
                  
data(:, 3) = reshape([1 1 1 1 0;
                      0 0 0 0 1;
                      0 1 1 1 0;
                      0 0 0 0 1;
                      1 1 1 1 0], 25, 1);
                  
data(:, 4) = reshape([0 0 0 1 0;
                      0 0 1 1 0;
                      0 1 0 1 0;
                      1 1 1 1 1;
                      0 0 0 1 0], 25, 1);
                  
data(:, 5) = reshape([1 1 1 1 1;
                      1 0 0 0 0;
                      1 1 1 1 0;
                      0 0 0 0 1;
                      1 1 1 1 0], 25, 1);
                  
test = reshape([0 1 1 1 1;
                0 1 0 0 0;
                0 1 1 1 0;
                0 0 0 1 0;
                1 1 1 1 0], 25, 1);
label = eye(5);
dataSet = {data, label};
[data, label, numOfBatch] = generateBatch(dataSet{1}, dataSet{2}, 1);

net = model({LinearLayer(25, 80), DropOut(80), ReLU(80), ...
             LinearLayer(80, 5), SoftMaxLayer(5)});
         
epoch = 1000;
printableLoss = [];

for i = 1 : epoch
    for j = 1 : numOfBatch
        out = net.forward(data{j});
        [loss, gradient] = Loss.CrossEntropy(label{j}, out);
        net.backward(gradient, 0.001, 0.8, 0.00);
    end
    printableLoss = [printableLoss mean(mean(loss))];
end
net2=net;
result = net2.forward(dataSet{1})
% result = net.forward(test)
plot(1 : epoch, printableLoss, 'r-');
