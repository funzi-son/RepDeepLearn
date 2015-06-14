function model = train_ae(conf)
% This code train a AE following Bengio 2006 paper: Greedy Layer-wise Training for Deep nets
% Authors: Son T

% Load data
trn_dat = get_data_from_file(conf.trn_dat_file,conf.row_dat);

[visNum,sz] = size(trn_dat);
hidNum      = conf.hidNum;
mmm = max(hidNum,visNum);
model.W = 1/(mmm)*(2*rand(visNum,hidNum)-1);
model.visB = zeros(visNum,1);
model.hidB = zeros(hidNum,1);

DW  = zeros(size(model.W));
DVB = zeros(size(model.visB));
DHB = zeros(size(model.hidB));

if isfield(conf,'v_unit'), v_unit = conf.v_unit; end
if isfield(conf,'h_unit'), h_unit = conf.h_unit; end
units

fprintf('Start training an AE: %d %s x %d %s\n',visNum,v_unit,hidNum,h_unit);



if conf.sNum ==0, conf.sNum = sz; end
    
if conf.bNum == 0
   bNum = sz/conf.sNum;
else
   bNum = conf.bNum;
end

ce_plot = [];
sparse_plot  = [];
for ei=1:conf.eNum
   %tic;
   inx = randperm(sz);
   lr = conf.params(1);
   centropy = 0;
   hspr = 0;
   for b=1:bNum
     visP = trn_dat(:,inx((b-1)*conf.sNum +1:min(b*conf.sNum,sz)));
     sNum = size(visP,2);
     % Upward pass
     hidI = bsxfun(@plus,model.W'*visP,model.hidB);
     hidP = vis2hid(hidI); 
     % Reconstruction
     visNI = bsxfun(@plus,model.W*hidP,model.visB);
     visN = vis2hid(visNI);
     
     vdiff = (visP-visN);         
     hdiff = (model.W'*vdiff).*hidP.*(1-hidP);
     wdiff = visP*hdiff' + vdiff*hidP';
     
    %pause
      DW = lr*(wdiff/sNum-conf.params(4)*model.W) + conf.params(3)*DW;
      model.W = model.W + DW;

      DVB = lr*mean(vdiff,2) + conf.params(3)*DVB;
      model.visB = model.visB +DVB;

      DHB = lr*mean(hdiff,2) + conf.params(3)*DHB;
      model.hidB  = model.hidB + DHB;

      % Sparsity
      if isfield(conf,'sparsity') && conf.lambda > 0
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
      centropy = centropy + cross_entropy(visP,visN);
      
   end
    %toc;
    fprintf('[Epoch %.4d]: Cross Entropy = %.5f| Sparsity = %.5f\n',ei,centropy,hspr);
    if isfield(conf,'plot')        
        ce_plot = [ce_plot, centropy];        
        sparse_plot = [sparse_plot, hspr];
    end
    %pause
end
if ~isempty('ce_plot')
    fig1 = figure(1);
    set(fig1,'Position',[10,20,300,200]);
    plot(1:size(ce_plot,2),log(ce_plot));
    xlabel('Epochs');ylabel('Cross Entropy');
end
end