function cut = cutting(imglist)
   cut=cell(0);
   n=20;
%ͼ���ֵ����ȥ��
   for i=1:length(imglist)
    I=imglist{i};
    img_gray=rgb2gray(I);
    img_bw=im2bw(I,graythresh(I));
    img=medfilt2(img_bw,[5,5]);
   %img=imfilter(img_bw,fspecial('gaussian',[9,9]));
    %se = strel('disk',1); 
   % I = imdilate(img,se);
   I=img;
    I=imresize(I,[n,4*n]);
%�ַ��ָ�  
    cut{i,1}=I(:,1:n);
    cut{i,2}=I(:,n+1:2*n);
    cut{i,3}=I(:,2*n+1:3*n);
    cut{i,4}=I(:,3*n+1:4*n);   %��J�д洢�ָ���ַ�
    end
end

