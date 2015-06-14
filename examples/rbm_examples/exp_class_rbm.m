function exp_class_rbm()
conf.hidNum    = 50;            % Number of hidden units
conf.eNum      = 10;             % Number of epoch
conf.bNum      = 0;              % Batch number, 0 means it will be decided by the number of training samples
conf.sNum      = 0;              % Number of samples in one batch
conf.gNum      = 1;              % Number of Gibbs sampling step
conf.params(1) = 0.5;            % Learning rate (starting)
conf.params(2) = conf.params(1); % This is unused
conf.params(3) = 0.01;           % Momentum
conf.params(4) = 0.00001;        % Weight decay

conf.sparsity  = 'EMIN';         % EMIN,KLMIN
conf.cumsparse = 1;              % Only for KLMIN, using the expectation of previous batches or not (see the code)
conf.sparse_w  = 1;              % Only for EMIN, apply sparsity to w or not(for Lee's approach)
conf.lambda    = 0;              % Sparsity penalty
conf.p         = 0.001;          % Sparsity constraint

conf.E_STOP_LR_REDUCE = 50;      % Number of bad updates before reduce the learning rate
conf.E_STOP = 5;                 % Number of learning rate decay before stop training

conf.gen  = 1;                   % Generative training 
                                 % Discriminative training = 1-conf.gen

conf.plot = 1;                  % Show training plots

conf.row_dat = 1;               % If one sample is a row, otherwise set to 0

TEST = 2;
%% Training RBMs
switch(TEST)
    case 1
        %for Yale
        trn_dat_file = 'yale_train_data_6';
        trn_lab_file = 'yale_train_label_6';
        tst_dat_file = 'yale_test_data_5';
        tst_lab_file = 'yale_test_label_5';
        conf.row = 32;
        conf.col = 32;
        conf.img_row_order = 0;
    case 2
        %for MNIST 
        trn_dat_file = 'mnist_train_dat_20k';
        trn_lab_file = 'mnist_train_lab_20k';
        tst_dat_file = 'mnist_test_dat_10k';
        tst_lab_file = 'mnist_test_lab_10k';
        conf.row = 28;
        conf.col = 28;
        conf.img_row_order = 1;
        conf.sNum = 100;
end
conf.trn_dat_file = trn_dat_file;
conf.trn_lab_file = trn_lab_file;
tic
model = class_rbm_train(conf);
toc

tst_dat = get_data_from_file(tst_dat_file,conf.row_dat);
tst_lab = get_data_from_file(tst_lab_file);
output = rbm_classify(model,tst_dat,2);
acc = sum((output==tst_lab+1))/size(tst_dat,2);
fprintf('[App 1] Test accuracy(RBM classification): %.5f\n',acc);

end

