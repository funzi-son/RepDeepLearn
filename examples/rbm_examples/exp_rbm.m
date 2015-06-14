function exp_rbm()
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
tic
model = gen_rbm_train(conf);
toc

% Visualize the basis vectors
visualize_1l_filters(model.W,100,conf.row,conf.col,'minmax',conf.img_row_order);

% APPLICATIONS

if isfield(conf,'v_unit'), v_unit = conf.v_unit; end
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
units;

trn_dat = get_data_from_file(trn_dat_file,conf.row_dat);
trn_labels = get_data_from_file(trn_lab_file,0)';
tst_dat = get_data_from_file(tst_dat_file,conf.row_dat);
tst_labels = get_data_from_file(tst_lab_file)';
%% Application 1: Feature extraction
fprintf('-----------------------------------------------------------\n');
fprintf('Starting Application 1: Classification using SVM with AEs features. Press any key to continue...\n');
pause;

trn_features = vis2hid(bsxfun(@plus,model.W'*trn_dat,model.hidB))'; 
tst_features = vis2hid(bsxfun(@plus,model.W'*tst_dat,model.hidB))'; 

% Train classifier
svmmod = svmtrain(trn_labels, trn_features,['-q -c 10 -g 0.01']); % Change value of c & g to get better result
[~, accuracy,~] = svmpredict(tst_labels, tst_features, svmmod);
tst_acc = accuracy(1);         
fprintf('[App 1] Test accuracy (AE features): %.5f\n',tst_acc);
%% Application 3: Denoising
% In this application, number of Gibbs sampling in the training is very important
fprintf('-----------------------------------------------------------\n');	
fprintf('Starting Application 2: Denoising impaired images. Press any key to continue...\n');
pause;

dev = 0.2;% Standard deviation of Gaussian noise
impaired = tst_dat+dev*randn(size(tst_dat));
hidS = vis2hid(bsxfun(@plus,model.W'*impaired,model.hidB));
%hidS = hidS>rand(size(hidS));
reconstructed  = hid2vis(bsxfun(@plus,model.W*hidS,model.visB));
MSE = sum(sum((tst_dat - reconstructed).^2))/numel(tst_dat);
fprintf('[App 2] Reconstruction Error %.5f\n',MSE);
figure();
subplot(1,3,1);show_images(tst_dat',100,conf.row,conf.col,conf.img_row_order);
subplot(1,3,2);show_images(impaired',100,conf.row,conf.col,conf.img_row_order);
subplot(1,3,3);show_images(reconstructed',100,conf.row,conf.col,conf.img_row_order);

end

