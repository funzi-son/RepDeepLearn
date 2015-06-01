function output = gen_rbm_classify(model,dat)
% classify by sampling
% sontran 2013
lNum = size(model.U,1);
sNum = size(dat,2);
labP = 0.5*ones(lNum,sNum);
hidP = logistic(bsxfun(@plus,model.W'*dat + model.U'*labP,model.hidB));
hidPs = hidP>rand(size(hidP));
hidNs = hidPs;
%% gibb sampling
%for g=1:gNum
    %visN = logistic(bsxfun(@plus,model.W*hidNs,model.visB));
    %visNs = visN>rand(size(visN));
    output = softmax_(bsxfun(@plus,model.U*hidNs,model.labB));
%    hidN = logistic(visNs*model.W + model.U(output,:) + repmat(model.hidB,sNum,1));
%    hidNs = hidN>rand(size(hidN));
%end 
end