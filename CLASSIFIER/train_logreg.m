function [model,vld_acc,tst_acc] = train_logreg(conf, trn_dat,trn_lab, vld_dat,vld_lab, tst_dat,tst_lab)

if nargin<2
trn_dat = get_data_from_file(conf.trn_dat_file,conf.row_dat);
if nargin<3
trn_lab = get_data_from_file(conf.trn_lab_file,conf.row_dat);
if nargin<4
vld_dat = get_data_from_file(conf.vld_dat_file,conf.row_dat);
if nargin<5
vld_lab = get_data_from_file(conf.vld_lab_file,conf.row_dat);
if nargin<6
tst_dat = get_data_from_file(conf.tst_dat_file,conf.row_dat);
if nargin<7
tst_lab = get_data_from_file(conf.tst_lab_file,conf.row_dat);
end
end
end
end
end
end

obj_fnc = conf.objective_function;
% Select cost function
objective_funcs;

% Select error function for update/back-prob
logreg_error_funcs;

% Get dimentionalities
[visNum,SZ] = size(trn_dat);
lNum = numel(unique(trn_lab));

% Batch 
if conf.sNum == 0, conf.sNum = SZ; end
if conf.bNum == 0, conf.bNum = ceil(SZ/conf.sNum); end

% Initialize params
model.W     = min([0.001 1/visNum])*(2*rand(visNum,lNum)-1);
model.labB  = zeros(lNum,1);

DW = zeros(size(model.W));
DB = zeros(size(model.labB));


running = 1; e=0; vld_best=0;
lr = conf.params(1);
while running
    e = e+1;
    inx = randperm(SZ);
    trn_acc = 0;
    cost_vl = 0;
    for b=1:conf.bNum
        iii = inx((b-1)*conf.sNum+1:min(b*conf.sNum,SZ));
        X = trn_dat(:,iii);        
        L = trn_lab(iii)+1;
        
        sNum = size(X,2);
        probs = get_probs(X,model);                                      
        
        % Target label
       lab =  discrete2softmax(L,lNum);

        err = error_fnc(probs,lab);
  
        diff = X*err'/sNum;
        % Update
        DW = lr*(diff-conf.params(4)*model.W) + conf.params(3)*DW;
        model.W = model.W + DW;
        
        DB =  lr*mean(err,2) + conf.params(3)*DB;
        model.labB = model.labB + DB;
        
        % Get error
        [~,predicts] = max(probs);
        trn_acc = trn_acc + sum(sum(L==predicts))/sNum; 
        % Get cost     
        cost_vl = cost_vl + cost_fnc(probs,lab);
    end
    trn_acc = trn_acc/conf.bNum;
    [~,predicts] = max(get_probs(vld_dat,model));
    vld_acc = sum(sum(vld_lab+1 == predicts))/numel(vld_lab); 
    [~,predicts] = max(get_probs(tst_dat,model));
    tst_acc = sum(sum(tst_lab+1 == predicts))/numel(tst_lab); 
    
    % Learning rate decay + early stopping
    
    
    % Print out
    fprintf('[Epoch %d] Cost=%.5f|trn_acc=%.5f|vld_acc=%.5f|tst_acc=%.5f\n',e,cost_vl/conf.bNum,trn_acc,vld_acc,tst_acc);


    % learning rate decay 
        if isfield(conf,'E_STOP_LR_REDUCE')
        if vld_acc<=vld_best
            acc_drop_count = acc_drop_count + 1;
            % If accuracy reduces for a number of time, then turn back to the
            % best model and reduce the learning rate
            if acc_drop_count > conf.E_STOP_LR_REDUCE
                fprintf('Learning rate reduced!\n');
                acc_drop_count = 0;
                es_count = es_count + 1; %number of reduce learning rate
                lr = lr/conf.LR_CHANGE_FACTOR;
                model = model_best;
            end
        else
            es_count = 0;
            acc_drop_count = 0;
            vld_best = vld_acc;
            tst_best = tst_acc;
            model_best = model;
        end
    end
    % Early stopping
    if isfield(conf,'E_STOP') 
        if isfield(conf,'desire_acc') && vld_acc >= conf.desire_acc, running=0;end
        
 
        if es_count > conf.E_STOP, running=0; end
    end

    % Check stop
    if e>=conf.eNum, running=0; end
end

if exist('model_best','var'), model = model_best; end
if exist('vld_best','var'), vld_acc = vld_best; end
if exist('model_best','var'), tst_acc = tst_best; end

end


function probs = get_probs(X,model)
    I = bsxfun(@plus,model.W'*X,model.labB);
    I = exp(bsxfun(@minus,I,max(I)));% exponents of normalized input
    probs = bsxfun(@rdivide,I,sum(I));
end

