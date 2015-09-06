function cc = eval_dbn(models,tst_dat,tst_lab)
% testing dbn
depth = size(models,2);
sz    = size(tst_dat,1);
for i=1:depth-1    
    tst_dat = logistic(tst_dat*models(i).W + repmat(models(i).hidB,sz,1));
    %trn_dat = trn_dat>rand(size(trn_dat));
end
    cc(1) = sum(tst_lab+1 == rbm_classify(models(depth),tst_dat,1))/sz;
    cc(2) = sum(tst_lab+1 == rbm_classify(models(depth),tst_dat,2))/sz;
end

