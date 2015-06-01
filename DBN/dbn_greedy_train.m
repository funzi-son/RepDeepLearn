function [vld_acc,tst_acc] = dbn_greedy_train(confs)
%% Greedy train RBM
%sontran2014
depth = size(confs,2);
for i=1:depth-1 
    if ~exist(confs(i).mod_f,'file')
        fprintf('Start training deep layer %d\n',i);
        rbm = gen_rbm_train(confs(i));
        save(confs(i).mod_f,'rbm');
        vis2hid_file(rbm,confs(i).trn_dat_file,confs(i).trn_out_file);
        if ~isempty(confs(i).vld_dat_file)
           vis2hid_file(rbm,confs(i).vld_dat_file,confs(i).vld_out_file);
        end
        if ~isempty(confs(i).tst_dat_file)
           vis2hid_file(rbm,confs(i).tst_dat_file,confs(i).tst_out_file);
        end
        clear rbm;
    end
end

if exist(confs(depth).mod_f,'file'), vld_acc = -1; tst_acc=-1; return; end
fprintf('Start training top layer');
[rbm,vld_acc,tst_acc] = class_rbm_train_rule(confs(depth));
save(confs(depth).mod_f,'rbm');
end