function cnn = cnnann(a,b)
     cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
    };
    cnn = cnnsetup(cnn, a, b);
% 学习率  
    opts.alpha = 2;  
    % 每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是  
    % 把所有样本都输入了，计算所有样本的误差了才调整一次权值  
    opts.batchsize = size(a, 3);   
    opts.numepochs =2000;
     cnn = cnntrain(cnn, a, b, opts);  % 如果是还未训练
end

