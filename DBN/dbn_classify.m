function cc = dbn_classify(model,tst_dat,tst_lab)
% testing dbn
depth = size(model,2);
sz    = size(tst_dat,1);
for i=1:depth-1       
    tst_dat = logistic(tst_dat*model(i).W + repmat(model(i).hidB,sz,1));
    %trn_dat = trn_dat>rand(size(trn_dat));
end
    cc(1) = sum(tst_lab+1 == rbm_classify(model(depth),tst_dat,1))/sz;
    cc(2) = sum(tst_lab+1 == rbm_classify(model(depth),tst_dat,2))/sz;
end

