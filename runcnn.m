%bp测试
function accuracy = runcnn(cnn,name,cut)
    rightnum = 0;
    sumnum = 0; 
   for i = 1 : length(name)
   img_name = name{i};
    for w=1:4
    input =cut{i,w};
     input_size = size(input);               %给出像素数
     result_input(:,:, w) = reshape(input',input_size(1,1),input_size(1,2));
    end
      cnn = cnnff(cnn, result_input);
      cnn.o;
      [~, mans] = max(cnn.o);
            img_name
            mans=mans-1
            sumnum = sumnum + 1;
      if (mans(1) == str2num(img_name(1))&mans(2) == str2num(img_name(2))&mans(3) == str2num(img_name(3))&mans(4) == str2num(img_name(4)))
          rightnum = rightnum + 1;
      end
    end
   % rightdata = [rightnum, sumnum-rightnum]
    %pie(rightdata, {'right', 'wrong'});
    accuracy=rightnum/sumnum ; 
    plot(cnn.rL); 
    xlabel('训练次数');
    ylabel('均方误差');
end