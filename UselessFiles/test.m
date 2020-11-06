clear
clc
a=3
for i = 1:a
    a=a+1
end













% a = reshape(1:16,4,4)'
% layer = AveragePoolLayer(1,2);
% out=layer.forward(a)
% back=layer.backward([1 3;5 7],0,0)














% ker = repmat(ones(2,2)/2^2, 1,1,1)
% img = [2,3;4,5];
% img = repmat(img, 1, 1, 2)
% out = kron(img,ker)















% layer = ConvLayer(1, 1, 2);
% data = [1 2 3;
%         4 5 6;
%         7 8 9];
% out = layer.forward(data);
% aim = filtern(data, [4 3;
%                      2 1], 'valid')
% for i = 1:1000
%     out = layer.forward(data)
%     delta = -(aim - out)
%     back = layer.backward(delta*0.0001, 0, 0)
% end
% out = layer.forward(data)















% data=imread('jzx.png');
% data=mean(data, 3);
% data=(data(1:2:499,:)+data(2:2:500,:))/2;
% data=(data(:,1:2:499)+data(:,2:2:500))/2;
% data=(data(1:2:249,:)+data(2:2:250,:))/2;
% data=(data(:,1:2:249)+data(:,2:2:250))/2;
% data=data/255;
% in=rand(125,125);
% aim=filtern(data, [0.0 0.2 0.0;
%                    0.0 0.2 0.0;
%                    0.0 0.2 0.0],'valid');
% aim=aim>0.5;
% aim=aim+0;
% % image(aim*255)
% net=model({ConvLayer(1,1,3),Sigmoid()});
% % out=net.forward(data);
% % image(out*256)
% for i=1:1000
%     out = net.forward(data);
% %     image(out*255)
% %     image(aim)
%     if (i==1)
% %         image(out*255);
%     end
%     [loss, gradient] = Loss.CrossEntropy(aim, out);
%     net.backward(gradient, 0.0001, 0.8, 0.001);
% end
% out=net.forward(data);
% image(out*255)













% img = reshape(1:9, 3, 3)';
% img = reshape(img, 3,3,1,1,1);
% ker = [1 2;3 4];
% 
% out = filtern(img,ker,'valid')
% 
% takeIn = [7 6;5 4];
% 
% g = filtern(img, takeIn, 'valid')
% 
% passback = convn(takeIn, ker, 'full')
% 
% layer = ConvLayer(1,1,2);
% layer_out = layer.forward(img)
% layer_back = layer.backward(takeIn, 0, 0)

















% img1_ch1 = reshape(1 : 9, 3, 3);
% img1_ch2 = reshape(10 : 18, 3, 3);
% img2_ch1 = reshape(19 : 27, 3, 3);
% img2_ch2 = reshape(28 : 36, 3, 3);
% 
% img = [img1_ch1 img1_ch2 img2_ch1 img2_ch2];
% img = reshape(img, 3, 3, 2, 1, 2);
% 
% ker1_ch1 = reshape(1 : 4, 2, 2);
% ker1_ch2 = reshape(5 : 8, 2, 2);
% ker2_ch1 = reshape(9 : 12, 2, 2);
% ker2_ch2 = reshape(13 : 16, 2, 2);
% 
% ker = [ker1_ch1 ker1_ch2 ker2_ch1 ker2_ch2];
% ker = reshape(ker, 2, 2, 2, 2, 1);
% 
% for i = 1 : 2
%     outputCache(:, :, 1, i, :)=filtern(img(:,:,:,1,:),ker(:,:,:,i,1),'valid');
% end
% out = outputCache;
% 
% layer = ConvLayer(2, 2, 2);
% layer_out = layer.forward(img);
% layer.backward(randn(2,2,2,2),0,0);











% a=[1 2;3 4]
% b=reshape(a,2,2,1)













% a=[1 2 3;4 5 6;7 8 9];
% b=[4 5;6 7];
% c=conv2(a,b,'full')
% d=conv2(b,a,'full')











% jzx = imread("jzx.png");
% fil = randn(3,3,3);
% new_jzx = convn(jzx, fil, 'valid');
% image(new_jzx);











% a=randn(5,5,3)
% b=randn(2,2,3)
% convn(a,b,valid)







% a=[1 2 3;4 5 6;7 8 9]
% b=[1 2;3 4]
% c=filtern(a, b, 'valid')












% img = randn(5, 5, 2, 2);
% layer = ConvLayer(2, 2, 4);
% originKer=layer.kernels
% for i = 1:1
%     layer.forward(img);
%     layer.backward(randn(2,2,4,2),0,0);
%     ker=layer.kernels
%     gker=layer.gKernels
% end











% img1_ch1 = reshape(1:9, 3, 3)';
% img1_ch2 = reshape(10:18, 3, 3)';
% img1(:,:,1) = img1_ch1;
% img1(:,:,2) = img1_ch2;
% 
% img2_ch1 = reshape(19:27, 3, 3)';
% img2_ch2 = reshape(28:36, 3, 3)';
% img2(:,:,1) = img2_ch1;
% img2(:,:,2) = img2_ch2;
% 
% img(:,:,:,1) = img1;
% img(:,:,:,2) = img2;
% 
% kernel = randn(2,2,2);
% kernel = rot90(kernel, 2)
% b(:,:,1,:) = kernel
% 
% 
% img1_conv = convn(img1, kernel, 'valid');
% img_conv = convn(img, kernel, 'valid')
% img_conv = convn(img, b, 'valid')
% a=squeeze(img_conv);









% a=reshape(1:9, 3, 3)'
% a1(:,:,1)=a
% a1(:,:,2)=a
% b=reshape(1:4,2, 2)'
% b1(:,:,1)=b
% b1(:,:,2)=b
% c=convn(a,b, 'valid')
% c1=convn(a1,b1, 'valid')












% img1_ch1=[1 2;3 4]
% img1_ch2=[5 6;7 8]
% img2_ch1=[9 10;11 12]
% img2_ch2=[13 14;15 16]
% img=[img1_ch1 img1_ch2 img2_ch1 img2_ch2]
% imgg=reshape(img, 2,2,2,2)
% f=[1 1 1 1 1;
%                       1 0 0 0 0;
%                       1 1 1 1 0;
%                       0 0 0 0 1;
%                       1 1 1 1 0]







% a=[1 2 3;
%    -1 -2 -3;]
% l0=(a<0)*0.1
% g0=(a>0)*1
% b=a.*l0+a.*g0










% model = ReLU(5)
% in=[1;2;0;-1;-2];
% model.forward(in)
% model.backward([1;3;0;-1;-2])








% currentOutput = [1; 3; 5]
% aiajMatrix2 = currentOutput * currentOutput'
% diag(currentOutput)

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