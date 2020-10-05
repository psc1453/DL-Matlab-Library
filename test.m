
currentOutput = [1; 3; 5]
aiajMatrix2 = currentOutput * currentOutput'
diag(currentOutput)

% a=[1;2;3;4]
% ai=repmat(a',4,1)
% aj=repmat(1-a,1,4)
% g=ai.*aj
% b=[1 2 3]
% diag(b)




% a=[1 2 3;4 5 6;7 8 9]
% s=size(a)
% e=eye(s)
% l=logical(e)
% a([1 0 0;0 1 0;0 0 1])












% clear;clc
% [x,y,z]=meshgrid(linspace(-2*pi,2*pi));
% G=@(x,y,z)x.*cos(y)+z.*y+sin(x+z);%记住用点运算
% isosurface(x,y,z,G(x,y,z),0)
% 
% 
% 
% 
% 
% 
% 
% 
% % dataSet = {[0 0 1 1;
% %             0 1 0 1;
% %             1 1 1 1], [0 0 1 1]};
% % [data, label, numOfBatch]=generateBatch(dataSet{1}, dataSet{2}, 4)
% % data{1}
% % label{1}
% 
% 
% 
% 
% 
% % clear
% % clc
% % 
% % data = [0 0 1;
% %         0 1 1;
% %         1 0 1;
% %         1 1 1]';
% % label = [0 1 1 0];
% % net = model({LinearLayer(3, 4),Sigmoid(4),LinearLayer(4, 1), Sigmoid(1)})
% % for i = 1 : 10000
% %     out = net.forward(data);
% %     loss = 2 * (label - out);
% %     net.backward(loss, 0.8);
% % end
% % out