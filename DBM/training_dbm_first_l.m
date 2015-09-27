function model = training_dbm_first_l(conf,trn_dat)
% Son Tran
%% Load data
if nargin<2
    trn_dat = get_data_from_file(conf.trn_dat_file);
end


%% initialization
visNum  = size(trn_dat,1);
hidNum  = conf.hidNum;
lr    = conf.params(1);
N     = conf.N;  % Number of epoch training with lr_1                     

model.W     = min(1/max(visNum,hidNum),0.001)*(2*rand(visNum,hidNum)-1);
model.visB  = zeros(visNum,1);
model.hidB  = zeros(hidNum,1);

DW    = zeros(size(model.W));
DVB   = zeros(visNum,1);
DHB   = zeros(hidNum,1);


%% ==================== Start training =========================== %%
for e=1:conf.eNum
    if e== N+1
        lr = conf.params(2);
    end
    err = 0;
    
    inx = randperm(SZ);
    
    for b=1:conf.bNum
       visP = trn_dat(:,inx((b-1)*conf.sNum+1:min(b*conf.sNum,Z)));
       sNum = size(visP,2);
       %up
       hidP = logistic(bsxfun(@plus,2*model.W'*visP,model.hidB));
       hidPs =  1*(hidP >rand(sNum,hidNum));
       hidNs = hidPs;
       for g=1:conf.gNum
           % down
           visN  = logistic(bsxfun(@plus,model.W*hidNs,model.visB));
           visNs = 1*(visN>rand(sNum,visNum));
           % up
           hidN  = logistic(bsxfun(@plus,2*model.W'*visNs,model.hidB));
           hidNs = 1*(hidN>rand(sNum,hidNum));
       end
       % Compute MSE for reconstruction
       err = err+ mse(visP,visN));
       % Update W,visB,hidB
       diff = 2*(visP*hidP' - visNs*hidN')/sNum;
       DW   = lr*(diff - conf.params(4)*model.W) +  conf.params(3)*DW;
       model.W    = model.W + DW;

       DVB  = lr*sum(2*visP - 2*visN,1)/sNum + conf.params(3)*DVB;
       model.visB = model.visB + DVB;

       DHB  = lr*sum(hidP - hidN,1)/sNum + conf.params(3)*DHB;
       model.hidB = model.hidB + DHB;
    end
    % Visualization
    if ~isempty(conf.vis_dir)        
        save_images(visN,strcat(conf.vis_dir,'1layer_rec_'),sNum,e,conf.row,conf.col);
    end
     fprintf('Epoch %d  : Error  = %f\n',e,err);
end
end