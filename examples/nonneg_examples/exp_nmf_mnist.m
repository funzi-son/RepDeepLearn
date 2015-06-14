function exp_nmf_mnist
X = get_data_from_file('mnist_train_dat_20k.mat',1);
conf.hidNum = 100;
conf.eNum = 100;
conf.stop = 0.001;
conf.costFnc = 'euclidean';%'i-divergence';% 'euclidean' or 'i-divergence'
conf.eps = 0.0000001;

conf.row = 28;
conf.col = 28;
conf.row_order = 1;
[~,W] = nmf(conf,X);
end