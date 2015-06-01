eval(strcat(mfilename,'_setting'));
X = get_data_from_file(strcat(DAT_DIR,'mnist_train_dat_10k.mat'))';
conf.hidNum = 100;
conf.eNum = 100;
conf.stop = 0.001;
conf.costFnc = 'i-divergence';% 'euclidean' or 'i-divergence'
conf.eps = 0.0000001;

conf.row = 28;
conf.col = 28;
conf.row_order = 1;
[~,W] = nmf(conf,X);