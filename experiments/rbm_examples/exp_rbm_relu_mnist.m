function exp_rbm_relu_mnist()
%EXP_RBM_RELU_MNIST Summary of this function goes here
%   Detailed explanation goes here

eval(strcat(mfilename,'_setting'));

fprintf('Number of experiments to be run is %d. Press a key to continue\n',...
    size(lrs,2)*size(mms,2)*size(csts,2));

pause(3);
for lr = lrs
for mm = mms
for cst = csts
conf.hidNum    = 500;  % Number of hidden units
conf.eNum      = 100;   % Number of epoch
conf.bNum      = 0;    % Batch number, 0 means it will be decided by the number of training samples
conf.sNum      = 100;  % Number of samples in one batch
conf.gNum      = 1;    % Number of Gibbs sampling
conf.params(1) = lr;  % Learning rate (starting)
conf.params(2) = conf.params(1); % This is unused
conf.params(3) = mm; % Momentum
conf.params(4) = cst; % Weight decay

conf.vis = 1;
conf.hid_unit  = 'RELU';

conf.vis       = 1;
%% Training RBMs
conf.trn_dat_file = strcat(DAT_DIR,'mnist_train_dat_20k_norm.mat');
EXP_DIR_ = strcat(EXP_DIR,'RELU',lm);
if ~exist(EXP_DIR_,'dir'), mkdir(EXP_DIR_); end
fname = strcat(EXP_DIR_,'rbm_h'...
    ,num2str(conf.hidNum),'_lr',num2str(conf.params(1)),'_mm',num2str(conf.params(3)),...
    '_cst',num2str(cst),'.mat');
iname = strrep(fname,'.mat','_img.bmp');
if exist(iname,'file'), continue; end

tic
model = gen_rbm_relu_train(conf);
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

