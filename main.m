clc;clear all;close all;
%% 训练集图片预处理
train=dir('E:\吴Qi An\课程资料\神经网络\验证码\train\*.jpg');         
train_list=cell(0);
for i=1:length(train)               
    rootpath=strcat('E:\吴Qi An\课程资料\神经网络\验证码\train','\',train(i).name);   
    train_list{i}=imread(rootpath);
end
%训练集图片预处理 二值化去噪和字符分割
train_cut=cutting(train_list);
%文件名预先标记为验证码的数字
for i = 1: length(train)
    str_name = train(i).name;
    train_name{i} = str_name(1:4); 
end
%% 测试集图片预处理
test=dir('E:\吴Qi An\课程资料\神经网络\验证码\test\*.jpg');
for i = 1: length(test)
    str_name = test(i).name;
    test_name{i} = str_name(1:4); 
    test_list{i}=imread(strcat('E:\吴Qi An\课程资料\神经网络\验证码\test','\',str_name));    
end   
test_cut=cutting(test_list);
%M=J{4,4};
%imshow(M)
%figure;
%imshow(imglist{4}); 
%%  bp 网络训练  
tic
V=double(rand(400,80));
W=double(rand(80,10));
delta_V=double(rand(400,80));
delta_W=double(rand(80,10));
yita=0.1;%学习率
yita1=0.05;%我自己加的参数，缩放激活函数的自变量防止输入过大进入函数的饱和区
%train_num=30;%训练样本中，每种数字多少张图，一共100张
x=double(zeros(1,400));%输入层
y=double(zeros(1,80));%中间层，也是隐藏层
output=double(zeros(1,10));%输出层
tar_output=double(zeros(1,10));%目标输出，即理想输出
delta=double(zeros(1,10));%一个中间变量，可以不管
imgs_sample=cell(10);
      imgs_sample_num=zeros(1,10);
     for i = 1 : length(train_name)
         img_name = train_name{i};  
        for j=1:4
           tmp_num=str2num(img_name(j))+1;
           imgs_sample_num(tmp_num)=imgs_sample_num(tmp_num)+1;
           imgs_sample{tmp_num, imgs_sample_num(tmp_num)} = train_cut{i,j};
        end
     end
%记录总的均方差便于画图
s_record=1:1000;
for train_control_num=1:1000  %训练次数控制
    s=0;
%读图，输入网络
    i = 1;
    for k = 1 : 10
        for j = 1 : imgs_sample_num(k)
            input = imgs_sample{k, j};
            input_size = numel(input);               %给出像素数
            inputs(i, :) = reshape(input',input_size,1); %input每行为一个字符，4*count行
            outputs(i, :) = zeros(10, 1);
            outputs(i,k) = 1;
tmp=inputs(i, :)';%图像二维矩阵转变为列向量，400维，作为输入
%计算输入层输入
x=double(tmp');%转化为行向量因为输入层X是行向量，并且化为浮点数
%得到隐层输入
y0=x*V;
%激活
y=1./(1+exp(-y0*yita1));
%得到输出层输入
output0=y*W;
output=1./(1+exp(-output0*yita1));
%计算预期输出
tar_output=double(zeros(1,10));
tar_output(k)=1.0;
%计算误差
%按照公式计算W和V的调整，为了避免使用for循环比较耗费时间，下面采用了直接矩阵乘法，更高效
delta=(tar_output-output).*output.*(1-output);
delta_W=yita*repmat(y',1,10).*repmat(delta,80,1);
tmp=sum((W.*repmat(delta,80,1))');
tmp=tmp.*y.*(1-y);
delta_V=yita*repmat(x',1,80).*repmat(tmp,400,1);
%计算均方差
s=s+sum((tar_output-output).*(tar_output-output));
%更新权值
W=W+delta_W;
V=V+delta_V; 
i = i + 1; 
   end
 end
s=s/2500 %不加分号，随时输出误差观看收敛情况
train_control_num           %不加分号，随时输出迭代次数观看运行状态
s_record(train_control_num)=s;%记录
end
plot(1:1000,s_record);
xlabel('训练次数');
   ylabel('均方误差');
   t1=toc
   tic
%% bp测试
correct_num=0;%记录正确的数量
number=1:4;
%测试集中，一共多少数字，9个，没有0
%test_num=100;%测试集中，每个数字多少个，最大100个
 %load W;%之前训练得到的W保存了，可以直接加载进来
 %load V;
 %load yita1;
imgs_sample=cell(10);
      imgs_sample_num=zeros(1,10);
     for i = 1 : length(test_name)
         img_name = test_name{i};  
        for j=1:4
           tmp_num=str2num(img_name(j))+1;
           imgs_sample_num(tmp_num)=imgs_sample_num(tmp_num)+1;
           imgs_sample{tmp_num, imgs_sample_num(tmp_num)} = test_cut{i,j};
        end
     end
%记录时间
for i = 1 : length(test_name)
   img_name = test_name{i};
    for w=1:4
    input =test_cut{i,w};
     input_size = numel(input);               %给出像素数
     result_input(w, :) = reshape(input',input_size,1);
tmp=result_input(w, :)';
tmp=tmp(:);
%计算输入层输入
x=double(tmp');
%得到隐层输入
y0=x*V;
%激活
y=1./(1+exp(-y0*yita1));
%得到输出层输入
o0=y*W;
o=1./(1+exp(-o0*yita1));
%最大的输出即是识别到的数字
[o,index]=sort(o);%对o每一列进行升序排序
number(w)=index(10)-1;
  end
    if number(1)==str2num(img_name(1))&&number(2)==str2num(img_name(2))&&number(3)==str2num(img_name(3))&&number(4)==str2num(img_name(4))
    correct_num=correct_num+1
end
img_name
number
end
correct_rate=correct_num/500;
sprintf('bp测试集识别准确率为%0.2f%%',correct_rate*100)
t2=toc
  %% cnn网络
  [a, b] = buildtrainset_cnn(train_name,train_cut);
   cnn = cnnann(a,b);
%load('-mat','E:\吴Qi An\课程资料\神经网络\验证码\cnn_save.mat') ;
    train_accuracy=runcnn(cnn,train_name,train_cut);
     test_accuracy=runcnn(cnn,test_name,test_cut);
     sprintf('cnn训练集识别准确率为%0.2f%%',train_accuracy*100)
      sprintf('cnn测试集识别准确率为%0.2f%%',test_accuracy*100)

     
  
    