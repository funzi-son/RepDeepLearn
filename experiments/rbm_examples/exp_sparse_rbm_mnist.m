function exp_sparse_rbm_mnist()
% Sparsity experiment on RBM
eval(strcat(mfilename,'_setting'));

fprintf('Number of experiments to be run is %d. Press a key to continue\n',...
    size(lrs,2)*size(mms,2)*size(csts,2)*size(lds,2)*size(ps,2));

pause(3);
for lr = lrs
for mm = mms
for cst = csts
for ld  = lds
for p = ps
conf.hidNum    = 200;  % Number of hidden units
conf.eNum      = 100;   % Number of epoch
conf.bNum      = 0;    % Batch number, 0 means it will be decided by the number of training samples
conf.sNum      = 100;  % Number of samples in one batch
conf.gNum      = 1;    % Number of Gibbs sampling
conf.params(1) = lr;  % Learning rate (starting)
conf.params(2) = conf.params(1); % This is unused
conf.params(3) = mm; % Momentum
conf.params(4) = cst; % Weight decay

conf.sparsity  = 'EMIN';% EMIN,KLMIN
conf.cumsparse = 1; % Only for KLMIN, using the expectation of previous batches or not (see the code)
conf.sparse_w  = 1; % Only for EMIN, apply sparsity to w or not(for Lee's approach)
conf.lambda    = ld;    % Sparsity penalty
conf.p         = p;% Sparsity constraint


conf.vis       = 1;
%% Training RBMs
conf.trn_dat_file = strcat(DAT_DIR,'mnist_train_dat_20k.mat');
EXP_DIR_ = strcat(EXP_DIR,conf.sparsity);
if ~exist(EXP_DIR_,'dir'), mkdir(EXP_DIR_); end
fname = strcat(EXP_DIR_,'rbm_h'...
    ,num2str(conf.hidNum),'_lr',num2str(conf.params(1)),'_mm',num2str(conf.params(3)),...
    '_cst',num2str(conf.params(4)),'_ld',num2str(conf.lambda),'_p',num2str(conf.p),'.mat');
iname = strrep(fname,'.mat','_img.bmp');
if exist(iname,'file'), continue; end

tic
model = gen_rbm_train(conf);
toc
% Visualize the basis vectors
imgs = model.W';
MN = min(min(imgs));
MX = max(max(imgs));
imgs = (imgs-MN)/(MX-MN);
save_images(imgs,100,28,28,iname);
end
end
end
end
end
