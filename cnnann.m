function cnn = cnnann(a,b)
     cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
    };
    cnn = cnnsetup(cnn, a, b);
% ѧϰ��  
    opts.alpha = 2;  
    % ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������  
    % �����������������ˣ�������������������˲ŵ���һ��Ȩֵ  
    opts.batchsize = size(a, 3);   
    opts.numepochs =2000;
     cnn = cnntrain(cnn, a, b, opts);  % ����ǻ�δѵ��
end

