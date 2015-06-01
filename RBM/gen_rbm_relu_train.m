function model = gen_rbm_relu_train(conf)
% Training RBMs with rectifier linear unit (Nair & Hinton)
% Son T- 2015

%% load file
dat = get_data_from_file(conf.trn_dat_file);
sum(sum(isnan(dat)))
%% setting up
hidNum = conf.hidNum;
[SZ,visNum] = size(dat);

model.W = 0.001*(2*randn(visNum,hidNum)-1);
model.U = [];

model.visB = zeros(1,visNum);
model.hidB = zeros(1,hidNum);
model.labB = [];

WD    = zeros(size(model.W));
visBD = zeros(size(model.visB));
hidBD = zeros(size(model.hidB));

bNum = conf.bNum;
if bNum==0, bNum=ceil(SZ/conf.sNum); end
%% running
lr = conf.params(1);
if ~isfield(conf,'N'), conf.N=10; end
for e=1:conf.eNum;
    inx = randperm(SZ);    
    
    res_e = 0;  % Reconstruction error
    hspr = 0;  % Sparsity
    
    if exist('conf.lr_decay','var'), lr=lr/conf.lr_decay^e;
    elseif e==conf.N+1, lr = conf.params(2); end
    
    for b=1:bNum
        iiii = inx((b-1)*conf.sNum+1:min(b*conf.sNum,SZ));
        visP = dat(iiii,:);
        sNum = size(visP,1);
        
        hidI = visP*model.W + repmat(model.hidB,sNum,1);
        hidP = max(0,hidI + rand(size(hidI)).*sqrt(logistic(hidI)));
        hidPs = hidP;
        hidNs = hidPs;
        %% gibb sampling
        for g=1:conf.gNum
            if conf.vis==1
                visN  = (hidNs*model.W' + repmat(model.visB,sNum,1));
                visNs = visN + randn(sNum,visNum);           
            else
                visN = logistic(bsxfun(@plus,hidNs*model.W',model.visB));
                visNs = visN>rand(size(visN));            
            end           
            
            hidI = visNs*model.W + repmat(model.hidB,sNum,1);                        
            hidN = max(0,hidI + rand(size(hidI)).*sqrt(logistic(hidI)));
            hidNs = hidN;
        end        
        res_e = res_e + sum(sqrt(sum((visP - visNs).^2,2)/visNum),1)/sNum;        

        %% updating        
        w_diff = (visP'*hidP - visNs'*hidN)/sNum;        
        WD = lr*(w_diff - conf.params(4)*model.W) + conf.params(3)*WD;
        model.W = model.W + WD;              
        visBD = lr*sum(visP - visNs,1)/sNum + conf.params(3)*visBD;
        model.visB  = model.visB + visBD;
        
        hidBD = lr*sum(hidPs - hidNs,1)/sNum + conf.params(3)*hidBD;
        model.hidB  = model.hidB + hidBD;              
        
    end
    fprintf('[Epoch %.3d] res_e = %.5f || spr_e = %.3f \n',e,res_e/bNum,hspr/bNum);
    if exist('conf.vis_dir','var')
        save_images(visN,100,28,28,strcat(conf.vis_dir,num2str(e),'.bmp'));
    end
end   
end

