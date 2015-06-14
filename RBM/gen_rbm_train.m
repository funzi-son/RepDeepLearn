function model = gen_rbm_train(conf)
% Train generative RBM with label
% Note: This one is new version, treating a data matrix as visNumxsNum
% sontran2013
%% load file
dat = get_data_from_file(conf.trn_dat_file);

if isfield(conf,'row_dat') && conf.row_dat, dat = dat'; end
%% setting up
hidNum = conf.hidNum;
[visNum,SZ] = size(dat);

model.W = (1/max(visNum,hidNum))*(2*rand(visNum,hidNum)-1);
model.U = [];

model.visB = zeros(visNum,1);
model.hidB = zeros(hidNum,1);
model.labB = [];

WD    = zeros(size(model.W));
visBD = zeros(size(model.visB));
hidBD = zeros(size(model.hidB));

if isfield(conf,'gpu') && conf.gpu
    dat = gpuArray(dat);
    model.W = gpuArray(model.W);
    model.visB = gpuArray(model.visB);
    model.hidB = gpuArray(model.hidB);
    WD = gpuArray(WD);
    visBD = gpuArray(visBD);
    hidBD = gpuArray(hidBD);
end
%% Define units for each layer
if isfield(conf,'v_unit'), v_unit = conf.v_unit; end
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
units
fprintf('Start training an RBM: %d %s x %d %s\n',visNum,v_unit,hidNum,h_unit);


%% Batch learning papameter
if conf.sNum == 0, conf.sNum = SZ; end
bNum = conf.bNum; if bNum==0, bNum=ceil(SZ/conf.sNum); end
%% running
lr = conf.params(1);
if ~isfield(conf,'N'), conf.N=10; end
for e=1:conf.eNum;
    inx = randperm(SZ);    
    
    res_e = 0;  % Reconstruction error
    hspr = 0;  % Sparsity
    
    if isfield(conf,'lr_decay'), lr=lr/conf.lr_decay^e;
    elseif e==conf.N+1, lr = conf.params(2); end
    
    for b=1:bNum
        iiii = inx((b-1)*conf.sNum+1:min(b*conf.sNum,SZ));
        visP = dat(:,iiii);
        sNum = size(visP,2);
                     
        hidP = vis2hid(bsxfun(@plus,model.W'*visP,model.hidB));
        hidPs = hid_sample(hidP);
        hidNs = hidPs;
        %% gibb sampling
        for g=1:conf.gNum              
            visN = hid2vis(bsxfun(@plus,model.W*hidNs,model.visB));
            visNs = vis_sample(visN);
            
            hidN = vis2hid(bsxfun(@plus,model.W'*visNs,model.hidB));
            hidNs = hid_sample(hidN);
        end        
        res_e = res_e + sum(sqrt(sum((visP - visNs).^2,1)/visNum))/sNum;

        %% updating        
        w_diff = (visP*hidP' - visNs*hidN')/sNum;        
        WD = lr*(w_diff - conf.params(4)*model.W) + conf.params(3)*WD;
        model.W = model.W + WD;              
       
        visBD = lr*sum(visP - visNs,2)/sNum + conf.params(3)*visBD;
        model.visB  = model.visB + visBD;
        
        hidBD = lr*sum(hidPs - hidNs,2)/sNum + conf.params(3)*hidBD;
        model.hidB  = model.hidB + hidBD;              
        %% Sparsity contrains
        if ~strcmp(h_unit,'relu') && isfield(conf,'sparsity') && conf.lambda>0
            if strcmp(conf.sparsity,'EMIN')
                expectation_min;
            elseif strcmp(conf.sparsity,'KLMIN')                
                kl_min;
            else
                fprintf('No sparsity constraint is set\n');
                continue;
            end            
           model.W = model.W + lr*w_diff;           
           model.hidB = model.hidB + lr*h_diff;
        end
    end
    fprintf('[Epoch %.3d] res_e = %.5f || spr_e = %.3f \n',e,res_e/bNum,hspr/bNum);
    if exist('conf.vis_dir','var')
        save_images(visN,100,conf.row,conf.col,strcat(conf.vis_dir,num2str(e),'.bmp'),conf.imrow_order);
    end
end
if isfield(conf,'gpu') && conf.gpu
        model.W = gather(model.W);
        model.visB = gather(model.visB);
        model.hidB = gather(model.hidB);
end
end