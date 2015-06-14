function exp_dbn()
trn_dat_file = 'mnist_train_dat_20k';
trn_lab_file = 'mnist_train_lab_20k';
vld_dat_file = 'mnist_vld_dat_10k';
vld_lab_file = 'mnist_vld_lab_10k';
tst_dat_file = 'mnist_test_dat_10k';
tst_lab_file = 'mnist_test_lab_10k';
% Directory to store feature file & model
EXP_MOD = '~/Documents/MOD/';
if ~exist(EXP_MOD,'dir'), mkdir(EXP_MOD); end
lm = '/'; % linux
% Learning setting for 1st layer
confs(1).hidNum = 200;
confs(1).eNum   = 50;
confs(1).bNum   = 0;
confs(1).sNum   = 100;
confs(1).gNum   = 1;
confs(1).params = [0.5 0.5 0.01 0.00001];
confs(1).N      = 10;
confs(1).row_dat = 1;
% Sparsity setting
confs(1).lambda = 0;

% File setting
DLAYER1 = strcat(EXP_MOD,'RBM1',lm);
if ~exist(DLAYER1,'dir'), mkdir(DLAYER1); end
file_prefixl1 = strcat('rbm1_h',num2str(confs(1).hidNum),'_lr',num2str(confs(1).params(1)),...
    '_mm',num2str(confs(1).params(3)),'_cst',num2str(confs(1).params(4)));
confs(1).trn_dat_file = trn_dat_file;
confs(1).trn_lab_file = trn_lab_file;
confs(1).vld_dat_file = vld_dat_file;
confs(1).vld_lab_file = vld_lab_file;
confs(1).tst_dat_file = tst_dat_file;
confs(1).tst_lab_file = tst_lab_file;

confs(1).trn_out_file = strcat(DLAYER1,file_prefixl1,'trn_out_dat.mat');
confs(1).vld_out_file = strcat(DLAYER1,file_prefixl1,'vld_out_dat.mat');
confs(1).tst_out_file = strcat(DLAYER1,file_prefixl1,'tst_out_dat.mat');


confs(1).mod_f  = strcat(DLAYER1,file_prefixl1,'.mat');

% Layer 2nd layer

confs(2).hidNum = 1000;
confs(2).eNum   = 50;
confs(2).bNum   = confs(1).bNum;
confs(2).sNum   = confs(1).sNum;
confs(2).gNum   = 1;
confs(2).params = [0.1 0.1 0.01 0.00001];
confs(2).N      = 10;
confs(2).row_dat = 0;
%Sparsity 
confs(2).lambda = 0;
%Classification setting
confs(2).class_type = 2; % 1: generative 2: discriminative
confs(2).E_STOP = 3;
confs(2).E_STOP_LR_REDUCE = 30;
confs(2).gen  = 1;
confs(2).dis = 1-confs(2).gen;

confs(2).val = 'model';
% File setting
DLAYER3 = strcat(EXP_MOD,'CRBM2',lm);
if ~exist(DLAYER3,'dir'), mkdir(DLAYER3); end

file_prefixl2 = strcat('crbm2_h',num2str(confs(2).hidNum),'_lr',num2str(confs(2).params(1)),...
    '_mm',num2str(confs(2).params(3)),'_cst',num2str(confs(2).params(3)),'_',file_prefixl1);

confs(2).trn_dat_file = confs(1).trn_out_file;
confs(2).trn_lab_file = confs(1).trn_lab_file;

confs(2).vld_dat_file = confs(1).vld_out_file;
confs(2).vld_lab_file = confs(1).vld_lab_file;

confs(2).tst_dat_file = confs(1).tst_out_file;
confs(2).tst_lab_file = confs(1).tst_lab_file;
confs(2).mod_f  = strcat(DLAYER3,file_prefixl2,'.mat');


confs(2).log_file = strcat(EXP_MOD,'log_',file_prefixl2,'.mat');


[vld_acc,tst_acc] = dbn_greedy_train(confs);

save(strcat(EXP_MOD,'DBN_',file_prefixl2,'.mat'),...
    'confs','vld_acc','tst_acc');

%% fine-tuning with back-prop
fprintf('===============================\nFine Tuning\n=========================\n');
for i=1:size(confs,2),
    ftconf.hidNum(i) = confs(i).hidNum;
    load(confs(i).mod_f);
    Ws{i} = rbm.W;
end
Ws{size(confs,2)+1} = rbm.U';

clearvars -except ftconf Ws trn_dat_file trn_lab_file vld_dat_file vld_lab_file tst_dat_file tst_lab_file;
ftconf.activationFnc = {'logsig','logsig','logsig'}; % Fixed, this should not be changed
ftconf.eNum   = 1000;
ftconf.bNum   = 0;  % Autormatically set, Number of batches (partitioned from all samples)
ftconf.sNum   = 100; % set later in each dataset: Number of sample in one batch
ftconf.params = [0.2 0.1 0.1 0.0001];
conf.E_STOP_LR_REDUCE = 50;      % Number of bad updates before reduce the learning rate
conf.E_STOP = 5;                 % Number of learning rate decay before stop training;


trn_dat = get_data_from_file(trn_dat_file,1);
trn_lab = get_data_from_file(trn_lab_file)';
vld_dat = get_data_from_file(vld_dat_file,1);
vld_lab = get_data_from_file(vld_lab_file)';
tst_dat = get_data_from_file(tst_dat_file,1);
tst_lab = get_data_from_file(tst_lab_file)';


% Learing rate decay & Early stopping for fine tuning
ftconf.E_STOP_LR_REDUCE = 50;
ftconf.E_STOP = 5;
model = train_nn(ftconf,Ws,[],trn_dat,trn_lab,vld_dat,vld_lab);


if exist('tst_dat','var')
   cout = run_nn(ftconf.activationFnc,model,tst_dat)';
   tst_acc = sum((cout-1)==tst_lab)/size(tst_lab,1);
   fprintf('Test accuracy = %.5f\n',tst_acc);
end
end

