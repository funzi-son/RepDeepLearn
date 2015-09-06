function exp_logreg()
addpath(genpath('~/Dropbox/WORK/DEEP_NET/'));
addpath(genpath('~/My.Academic/DATA/CHAR.REG/MNIST/'));
% Learning params
conf.objective_function = 'cent';%'mse';%'mle';
conf.eNum   = 1000;
conf.bNum   = 0;
conf.sNum   = 100;
conf.params = [0.5 0.5 0.01 0.00001];
% Early stopping
conf.E_STOP_LR_REDUCE = 50;      % Number of bad updates before reduce the learning rate
conf.E_STOP = 5;                 % Number of learning rate decay before stop training;

% Data
conf.trn_dat_file = 'mnist_train_dat_50k.mat';
conf.trn_lab_file = 'mnist_train_lab_50k.mat';
conf.vld_dat_file = 'mnist_vld_dat_10k.mat';
conf.vld_lab_file = 'mnist_vld_lab_10k.mat';
conf.tst_dat_file = 'mnist_test_dat_10k.mat';
conf.tst_lab_file = 'mnist_test_lab_10k.mat';

conf.row_dat = 1;

% Training & testing
[~,vld_acc,tst_acc] = train_logreg(conf);

end

