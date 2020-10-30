clear
clc

mnist = load('MNISTData.mat');
train_label = mnist.D_Train(:, 1:60000);
train_data(:, :, 1, :) = mnist.X_Train(:, :, 1:60000);
test_label = mnist.D_Test(:,1:100);
test_data(:, :, 1, :) = mnist.X_Test(:,:,1:100);
samples=size(train_data, 4);
batchSize=20;
iters=samples/batchSize;
net = model({ConvLayer(1, 20, 9), ReLU(), AveragePoolLayer(20, 2), ShapeChange([10,10,20], [2000]), LinearLayer(2000, 100), Sigmoid(), LinearLayer(100, 10), SoftMaxLayer()});
printableLoss=[];
acc=[];
for j=1:1
    for i = 1:iters
        out = net.forward(train_data(:,:,:,i:iters:end-iters+i));
        [loss, gradient] = Loss.CrossEntropy(train_label(:,i:iters:end-iters+i), out);
        net.backward(gradient, 0.05, 0.8, 0.0005);
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
out = net.forward(test_data1);
[~,aim_idx1]=max(test_label1);
[~,out_idx1]=max(out);
error1=sum(aim_idx1==out_idx1);
fprintf('%f\n',error1/length(out_idx1));
plot(printableLoss)
% hold on
% plot(acc)

