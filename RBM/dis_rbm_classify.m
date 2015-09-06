function output = dis_rbm_classify(model,dat)
% Classify by max energy
% dat: nxm (m is number of samples)
% sontran2013
SZ     = size(dat,2);
num    = 1000; % Increase this value according to the GPU to improve performance
bNum   = ceil(SZ/num);
hidNum = size(model.W,2);
lNum   = size(model.U,1);
%size(repmat(dat*model.W + repmat(model.hidB,sNum,1),[1,1,lNum]))
%size(repmat(reshape((eye(lNum)*model.U)',[1 hidNum lNum]),[sNum,1,1])) 
output = zeros(1,SZ);
for b=1:bNum
    sInx = (b-1)*num+1;
    eInx = min(b*num,SZ);
    ddd = dat(:,sInx:eInx);
    sNum = size(ddd,2);
    xxxx = repmat(model.W'*ddd + repmat(model.hidB,1,sNum),[1,1,lNum]) + ...
        repmat(reshape((model.U'*eye(lNum)),[hidNum,1,lNum]),[1,sNum,1]);   
    xxxx = reshape(sum(log(1+exp(xxxx)),1),[sNum lNum]);
    [~,outp] = max(xxxx,[],2);    
    output(sInx:eInx) = outp';
end
end