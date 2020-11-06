clear
clc

% Add necessarry directories to path
addpath('Tools', 'Framework', 'Layers');

% Load dataset
mnist = load('MNISTData.mat');
train_label = mnist.D_Train;
train_data(:, :, 1, :) = mnist.X_Train;
test_label = mnist.D_Test(:, 1:100);
test_data(:, :, 1, :) = mnist.X_Test(:, :, 1:100);
% Count sample number
samples=size(train_data, 4);

% Training parameters
batchSize = 1;
learningRate = 0.005;
momentum = 0;
l2 = 0;

% Model definition
net = model({ConvLayer(1, 10, 9), ReLU(1), AveragePoolLayer(10, 2), ...
             ShapeChange([10,10,10], [1000]), DropOut(0.3), ...
             FullConnectLayer(1000, 100), ReLU(1), FullConnectLayer(100, 10), SoftMaxLayer()});
         
% Plot list
loss=[];
accuracy=[];

% Train
iters=samples/batchSize;
net.trainMode();
for j=1:2
    for i = 1:iters
        out = net.forward(train_data(:,:,:,i:iters:end-iters+i));
        [currentLoss, gradient] = Loss.CrossEntropy(train_label(:,i:iters:end-iters+i), out);
        net.backward(gradient, learningRate, momentum, l2);
        loss=[loss mean(mean(currentLoss))];
        
%         out = net.forward(test_data);
%         [~,aim_idx]=max(test_label);
%         [~,out_idx]=max(out);
%         error=sum(aim_idx==out_idx);
%         acc = [acc error/length(out_idx)];
    end
end

% Test
test_label1 = mnist.D_Test(:,1:10000);
test_data1(:, :, 1, :) = mnist.X_Test(:,:,1:10000);
net.TestMode();
out = net.forward(test_data1);
[~,aim_idx1]=max(test_label1);
[~,out_idx1]=max(out);
error1=sum(aim_idx1==out_idx1);
fprintf('%f\n',error1/length(out_idx1));
plot(loss)
% hold on
% plot(acc)

