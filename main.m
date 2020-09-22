clear
clc
epoch = 10000;
dataSet = {[0 0 1;
            0 1 1;
            1 0 1;
            1 1 1], [0; 0; 1; 1]};
[data, label, numOfBatch]=generateBatch(dataSet{1}, dataSet{2}, 2);
pr=[];
model1 = model({LinearLayer(3, 1), Sigmoid(1)});
for i = 1 : epoch
    out = model1.forward([1;
                          0;
                          1]);
    model1.backward();
    loss = LinearLoss([0], out);
    pr=[pr loss.^2];
    model1.step(0.8*loss);
end
a=model1.forward([1;
                  0;
                  1])
plot(1:10000, pr,'b-');
