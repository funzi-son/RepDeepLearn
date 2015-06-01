function exp_drbm_mnist()
%EXP_DRBM_MNIST Summary of this function goes here
%   Detailed explanation goes here
eval(strcat(mfilename,'_setting'));

for mm = mms
for cst = csts
conf.hidNum = 500;
conf.eNum   = 2000;
conf.bNum   = 10;
conf.sNum   = 100;
conf.gNum   = 1;
conf.params = [0.5 0.5 mm cst];
conf.N      = 10;
conf.row_dat=1;
% GPU
conf.gpu    = 0;
%Sparsity 
conf.lambda = 0;
%Classification setting
conf.class_type = 2; % 1: generative 2: discriminative
conf.E_STOP_LR_REDUCE = 50;
conf.E_STOP = 5;
conf.gen  = 1;
conf.dis = 1-conf.gen;

conf.trn_dat_file = strcat(DAT_DIR,'mnist_train_dat_60k.mat');
conf.trn_lab_file = strcat(DAT_DIR,'mnist_train_lab_60k.mat');

conf.vld_dat_file = strcat(DAT_DIR,'mnist_test_dat_10k.mat');
conf.vld_lab_file = strcat(DAT_DIR,'mnist_test_lab_10k.mat');

conf.tst_dat_file = '';%strcat(DAT_DIR,'mnist_test_dat_10k.mat');
conf.tst_lab_file = '';%strcat(DAT_DIR,'mnist_test_lab_10k.mat');

conf.mod_f = strcat(EXP_DIR,'rbm_h',num2str(conf.hidNum),'_lr',num2str(conf.params(1))...
                           ,'_mm',num2str(conf.params(3)),'_cst',num2str(conf.params(4)),'.mat');
conf.log_file = strrep(conf.mod_f,'rbm_','log_');

if exist(conf.mod_f,'file'), continue; end
model = class_rbm_train(conf);


if isfield(conf,'gpu') && conf.gpu
    model.W    = gather(model.W);
    model.U    = gather(model.U);

    model.visB = gather(model.visB);
    model.hidB = gather(model.hidB);
    model.labB = gather(model.labB);
end
save(conf.mod_f,'model');
end
end
end

