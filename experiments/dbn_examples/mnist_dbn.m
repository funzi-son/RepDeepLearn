function mnist_dbn()
eval(strcat(mfilename,'_setting'));

%trn_dat_file = 'mnist_5k_trn_dat.mat';
%trn_lab_file = 'mnist_5k_trn_lab.mat';
trn_dat_file = 'mnist_train_dat_10k.mat';

for trial=2:4
for l1r = l1rs
for mm1 = mm1s
for cst1 = cst1s
% Learning setting for 1st layer
confs(1).hidNum = 200;
confs(1).eNum   = 50;
confs(1).bNum   = 0;
confs(1).sNum   = 100;
confs(1).gNum   = 1;
confs(1).params = [l1r l1r mm1 cst1];
confs(1).N      = 10;
confs(1).row_dat = 1;
% Sparsity setting
confs(1).lambda = 0;

% File setting
DLAYER1 = strcat(EXP_MOD,'RBM1',lm);
if ~exist(DLAYER1,'dir'), mkdir(DLAYER1); end
file_prefixl1 = strcat('rbm1_h',num2str(confs(1).hidNum),'_lr',num2str(l1r),'_mm',num2str(mm1),'_cst',num2str(cst1),'_trial',num2str(trial));
confs(1).trn_dat_file = strcat(DAT_DIR,trn_dat_file);
confs(1).trn_out_file = strcat(DLAYER1,file_prefixl1,'trn_out_dat.mat');
confs(1).trn_rule_out_file = strcat(DLAYER1,file_prefixl1,'trn_rule_dat.mat');

confs(1).mod_f  = strcat(DLAYER1,file_prefixl1,'.mat');

for l2r = l2rs
for mm2 = mm2s
for cst2 = cst2s

confs(2).hidNum = 1000;
confs(2).eNum   = 500;
confs(2).bNum   = confs(1).bNum;
confs(2).sNum   = confs(1).sNum;
confs(2).gNum   = 1;
confs(2).params = [l2r l2r mm2 cst2];
confs(2).N      = 10;
confs(2).row_dat = 0;
%Sparsity 
confs(2).lambda = 0;
%Classification setting
confs(2).class_type = 2; % 1: generative 2: discriminative
confs(2).E_STOP = 3;
confs(2).E_STOP_LR_REDUCE = 30;
confs(2).desire_acc = 0.975;
confs(2).gen  = 1;
confs(2).dis = 1-confs(2).gen;

confs(2).val = 'model';
% File setting
DLAYER3 = strcat(EXP_MOD,'CRBM2',lm);
if ~exist(DLAYER3,'dir'), mkdir(DLAYER3); end

file_prefixl2 = strcat('crbm2_h',num2str(confs(2).hidNum),'_lr',num2str(l2r),'_mm',num2str(mm2),'_cst',num2str(cst2),'_',file_prefixl1);

confs(2).trn_dat_file = confs(1).trn_out_file;
confs(2).trn_rule_file = confs(1).trn_rule_out_file;
confs(2).trn_lab_file = confs(1).trn_lab_file;

confs(2).vld_dat_file = confs(1).vld_out_file;
confs(2).vld_rule_file = confs(1).vld_rule_out_file;
confs(2).vld_lab_file = confs(1).vld_lab_file;

confs(2).tst_dat_file = confs(1).tst_out_file;
confs(2).tst_rule_file = confs(1).tst_rule_out_file;
confs(2).tst_lab_file = confs(1).tst_lab_file;
confs(2).mod_f  = strcat(DLAYER3,file_prefixl2,'.mat');


confs(2).log_file = strcat(EXP_MOD,'log_',file_prefixl2,'.mat');


if exist(confs(2).mod_f,'file'), continue; end

[vld_acc,tst_acc] = dbn_rule_greedy_train(confs);

save(strcat(EXP_MOD,'DBN_',file_prefixl2,'.mat'),...
    'confs','vld_acc','tst_acc');

end % cst2
end % mm2
end % l2r

end % cst1
end % mm1
end % l1r

end % trial

end

