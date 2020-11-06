clear
clc

addpath('Tools', 'Framework', 'Layers');

mnist = load('MNISTData.mat');
train_label = mnist.D_Train(:, 1:6000);
train_data(:, :, 1, :) = mnist.X_Train(:, :, 1:6000);
test_label = mnist.D_Test(:,1:100);
test_data(:, :, 1, :) = mnist.X_Test(:,:,1:100);
samples=size(train_data, 4);
batchSize=1;
iters=samples/batchSize;
net = model({ConvLayer(1, 10, 9), ReLU(1), AveragePoolLayer(10, 2), ...
             ShapeChange([10,10,10], [1000]), DropOut(0.3), ...
             LinearLayer(1000, 100), ReLU(1), LinearLayer(100, 10), SoftMaxLayer()});
printableLoss=[];
acc=[];
for j=1:1
    for i = 1:iters
        out = net.forward(train_data(:,:,:,i:iters:end-iters+i));
        [loss, gradient] = Loss.CrossEntropy(train_label(:,i:iters:end-iters+i), out);
        net.backward(gradient, 0.005, 0.0, 0.0000);
        printableLoss=[printableLoss mean(mean(loss))];
        
%         out = net.forward(test_data);
%         [~,aim_idx]=max(test_label);
%         [~,out_idx]=max(out);
%         error=sum(aim_idx==out_idx);
%         acc = [acc error/length(out_idx)];
    end
end
test_label1 = mnist.D_Test(:,1:10000);
test_data1(:, :, 1, :) = mnist.X_Test(:,:,1:10000);
net.TestMode();
out = net.forward(test_data1);
[~,aim_idx1]=max(test_label1);
[~,out_idx1]=max(out);
error1=sum(aim_idx1==out_idx1);
fprintf('%f\n',error1/length(out_idx1));
plot(printableLoss)
% hold on
% plot(acc)

