clc;clear all;close all;
%% ѵ����ͼƬԤ����
train=dir('E:\��Qi An\�γ�����\������\��֤��\train\*.jpg');         
train_list=cell(0);
for i=1:length(train)               
    rootpath=strcat('E:\��Qi An\�γ�����\������\��֤��\train','\',train(i).name);   
    train_list{i}=imread(rootpath);
end
%ѵ����ͼƬԤ���� ��ֵ��ȥ����ַ��ָ�
train_cut=cutting(train_list);
%�ļ���Ԥ�ȱ��Ϊ��֤�������
for i = 1: length(train)
    str_name = train(i).name;
    train_name{i} = str_name(1:4); 
end
%% ���Լ�ͼƬԤ����
test=dir('E:\��Qi An\�γ�����\������\��֤��\test\*.jpg');
for i = 1: length(test)
    str_name = test(i).name;
    test_name{i} = str_name(1:4); 
    test_list{i}=imread(strcat('E:\��Qi An\�γ�����\������\��֤��\test','\',str_name));    
end   
test_cut=cutting(test_list);
%M=J{4,4};
%imshow(M)
%figure;
%imshow(imglist{4}); 
%%  bp ����ѵ��  
tic
V=double(rand(400,80));
W=double(rand(80,10));
delta_V=double(rand(400,80));
delta_W=double(rand(80,10));
yita=0.1;%ѧϰ��
yita1=0.05;%���Լ��ӵĲ��������ż�������Ա�����ֹ���������뺯���ı�����
%train_num=30;%ѵ�������У�ÿ�����ֶ�����ͼ��һ��100��
x=double(zeros(1,400));%�����
y=double(zeros(1,80));%�м�㣬Ҳ�����ز�
output=double(zeros(1,10));%�����
tar_output=double(zeros(1,10));%Ŀ����������������
delta=double(zeros(1,10));%һ���м���������Բ���
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
%��¼�ܵľ�������ڻ�ͼ
s_record=1:1000;
for train_control_num=1:1000  %ѵ����������
    s=0;
%��ͼ����������
    i = 1;
    for k = 1 : 10
        for j = 1 : imgs_sample_num(k)
            input = imgs_sample{k, j};
            input_size = numel(input);               %����������
            inputs(i, :) = reshape(input',input_size,1); %inputÿ��Ϊһ���ַ���4*count��
            outputs(i, :) = zeros(10, 1);
            outputs(i,k) = 1;
tmp=inputs(i, :)';%ͼ���ά����ת��Ϊ��������400ά����Ϊ����
%�������������
x=double(tmp');%ת��Ϊ��������Ϊ�����X�������������һ�Ϊ������
%�õ���������
y0=x*V;
%����
y=1./(1+exp(-y0*yita1));
%�õ����������
output0=y*W;
output=1./(1+exp(-output0*yita1));
%����Ԥ�����
tar_output=double(zeros(1,10));
tar_output(k)=1.0;
%�������
%���չ�ʽ����W��V�ĵ�����Ϊ�˱���ʹ��forѭ���ȽϺķ�ʱ�䣬���������ֱ�Ӿ���˷�������Ч
delta=(tar_output-output).*output.*(1-output);
delta_W=yita*repmat(y',1,10).*repmat(delta,80,1);
tmp=sum((W.*repmat(delta,80,1))');
tmp=tmp.*y.*(1-y);
delta_V=yita*repmat(x',1,80).*repmat(tmp,400,1);
%���������
s=s+sum((tar_output-output).*(tar_output-output));
%����Ȩֵ
W=W+delta_W;
V=V+delta_V; 
i = i + 1; 
   end
 end
s=s/2500 %���ӷֺţ���ʱ������ۿ��������
train_control_num           %���ӷֺţ���ʱ������������ۿ�����״̬
s_record(train_control_num)=s;%��¼
end
plot(1:1000,s_record);
xlabel('ѵ������');
   ylabel('�������');
   t1=toc
   tic
%% bp����
correct_num=0;%��¼��ȷ������
number=1:4;
%���Լ��У�һ���������֣�9����û��0
%test_num=100;%���Լ��У�ÿ�����ֶ��ٸ������100��
 %load W;%֮ǰѵ���õ���W�����ˣ�����ֱ�Ӽ��ؽ���
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
%��¼ʱ��
for i = 1 : length(test_name)
   img_name = test_name{i};
    for w=1:4
    input =test_cut{i,w};
     input_size = numel(input);               %����������
     result_input(w, :) = reshape(input',input_size,1);
tmp=result_input(w, :)';
tmp=tmp(:);
%�������������
x=double(tmp');
%�õ���������
y0=x*V;
%����
y=1./(1+exp(-y0*yita1));
%�õ����������
o0=y*W;
o=1./(1+exp(-o0*yita1));
%�����������ʶ�𵽵�����
[o,index]=sort(o);%��oÿһ�н�����������
number(w)=index(10)-1;
  end
    if number(1)==str2num(img_name(1))&&number(2)==str2num(img_name(2))&&number(3)==str2num(img_name(3))&&number(4)==str2num(img_name(4))
    correct_num=correct_num+1
end
img_name
number
end
correct_rate=correct_num/500;
sprintf('bp���Լ�ʶ��׼ȷ��Ϊ%0.2f%%',correct_rate*100)
t2=toc
  %% cnn����
  [a, b] = buildtrainset_cnn(train_name,train_cut);
   cnn = cnnann(a,b);
%load('-mat','E:\��Qi An\�γ�����\������\��֤��\cnn_save.mat') ;
    train_accuracy=runcnn(cnn,train_name,train_cut);
     test_accuracy=runcnn(cnn,test_name,test_cut);
     sprintf('cnnѵ����ʶ��׼ȷ��Ϊ%0.2f%%',train_accuracy*100)
      sprintf('cnn���Լ�ʶ��׼ȷ��Ϊ%0.2f%%',test_accuracy*100)

     
  
    