function all_exp_gen_rbm()
% Generative RBM 
% Son T
eval(strcat(mfilename,'_setting'));
fprintf('Number of experiments to be run is %d. Press a key to continue\n',...
    size(lrs,2)*size(mms,2)*size(csts,2)*size(lds,2)*size(ps,2));

conf.hidNum    = 500;  % Number of hidden units
conf.eNum      = 100;   % Number of epoch
conf.bNum      = 0;    % Batch number, 0 means it will be decided by the number of training samples
conf.sNum      = 100;  % Number of samples in one batch
conf.gNum      = 1;    % Number of Gibbs sampling

conf.v_unit    = 'binary';
conf.h_unit    = 'binary';

if DAT_CASE==1
    dat_file = strcat(DAT_DIR,'mnist_train_dat_20k.mat');    
    conf.row = 28;
    conf.col = 28;
    conf.row_dat = 1; % one data point is one row
    conf.imrow_order = 1;  % The images are vectorized row-by-row
elseif DAT_CASE==2
    dat_file = strcat(DAT_DIR,'yale_train_data_9.mat');
    conf.row = 32;
    conf.col = 32;
    conf.row_dat = 1; % one data point is one row
    conf.imrow_order = 0; % The images are vectorized col-by-col
end
if strcmp(conf.v_unit,'gaussian')       
       norm = 'mean-var';       
       fznorm
       data = norm_fnc(get_data_from_file(dat_file),0,1);       
       dat_file = strrep(dat_file,'.mat','_norm.mat');       
       if ~exist(dat_file,'file'), save(dat_file,'data'); end
       clear data;
end
pause(3);
for lr = lrs
for mm = mms
for cst = csts
conf.params(1) = lr;  % Learning rate (starting)
conf.params(2) = conf.params(1); % This is unused
conf.params(3) = mm; % Momentum
conf.params(4) = cst; % Weight decay

if strcmp(SPARSITY,'RELU'), conf.h_unit  = 'relu'; lds = [0]; ps=[0]; end
for ld = lds
for p = ps

conf.sparsity  = SPARSITY;% EMIN,KLMIN
conf.cumsparse = 1; % Only for KLMIN, using the expectation of previous batches or not (see the code)
conf.sparse_w  = 1; % Only for EMIN, apply sparsity to w or not(for Lee's approach)
conf.lambda    = ld;    % Sparsity penalty
conf.p         = p;% Sparsity constraint


conf.vis       = 1;
%% Training RBMs
conf.trn_dat_file = dat_file;

EXP_DIR_ = strcat(EXP_DIR,SPARSITY,lm);
if ~exist(EXP_DIR_,'dir'), mkdir(EXP_DIR_); end
fname = strcat(EXP_DIR_,'rbm_h'...
    ,num2str(conf.hidNum),'_lr',num2str(conf.params(1)),'_mm',num2str(conf.params(3)),...
    '_cst',num2str(cst),'.mat');
iname = strrep(fname,'.mat','_img.bmp');
if exist(iname,'file'), continue; end

tic
model = gen_rbm_train(conf);
%model =  gen_rbm_relu_train(conf);
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

end

