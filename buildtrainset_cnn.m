%创建训练集
%分割后图像转为1维向量
function [inputs,outputs] = buildtrainset_cnn(train_name,train_cut);
    imgs_sample=cell(10);
      imgs_sample_num=zeros(1,10);
     for i = 1 : length(train_name)
         img_name = train_name{i};  
        for j=1:4; 
           tmp_num=str2num(img_name(j))+1;
           imgs_sample_num(tmp_num)=imgs_sample_num(tmp_num)+1;
           imgs_sample{tmp_num, imgs_sample_num(tmp_num)} = train_cut{i,j};
        end
     end
    i = 1;
    for k = 1 : 10
        for j = 1 : imgs_sample_num(k)
            input = imgs_sample{k, j};
            input_size = size(input);               
            inputs(:,:,i) = reshape(input',input_size(1,1),input_size(1,2)); %input每行为一个字符，4*count行
            outputs(:,i) = zeros(10, 1);
            outputs(k,i) = 1;
            i = i + 1;
        end
    end
end
