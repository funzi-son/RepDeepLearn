function model = training_dbm_mf_all(model,trn_dat)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%     Training Deep Boltzmann Machine using mean-field posterior appox 
%     and persistence CD
%     sontran2013
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % 

%% Load data
if nargin<2
    trn_dat = get_data_from_file(conf.trn_dat_file);
end

depth = size(model,2);
SZ     = size(trn_dat,2);
visNum = size(trn_dat,1);

sNum   = conf.sNum;
M      = conf.M;       

% Initialize params
DW    =  cell(depth,1);
DVB   =  zeros(visNum,1);
DHB1  =  cell(depth,1);

for i=1:depth
end

% Initialize fantasy particle
v_f  = round(rand(M,visNum));
h1_f = logistic(v_f*(2*Ws{1}) + repmat(hidBs{1},M,1));
h2_f = rand(M,conf.hidNum2);

lr = conf.lr;
h = plot(nan);
for e = 1:conf.eNum    
    inx = randperm(all_s);       
    re = 0;
    for b=1:conf.bNum
        lr = lr/(1.000015);
        X = trn_dat(inx((b-1)*sNum+1:b*sNum),:);
        % approximate posterior distribution
        converged = 0;
        count= 1;
        h1_mu = logistic(X*(2*Ws{1}) + repmat(hidBs{1},sNum,1));
        h2_mu = logistic(h1_mu*Ws{2} + repmat(hidBs{2},sNum,1));
        while ~converged
            h1_mu = logistic(X*Ws{1} + h2_mu*Ws{2}' + repmat(hidBs{1},sNum,1));
            h2_mu = logistic(h1_mu*Ws{2} + repmat(hidBs{2},sNum,1));
            %converging condition 1 (max_iter is reach)
            if count >= conf.max_iter
                converged = 1;
                fprintf('[DBM_GEN] Max_iter has been reached. Optimal mean-field may not be found\n');
            end
            %converging condition 2 (derivatives is close to 0)
            if count>1
                %disp([sum(sum(abs(o_h1_mu-h1_mu))) sum(sum(abs(o_h2_mu-h2_mu)))]);
                if sum(sum(abs(o_h1_mu-h1_mu),2))/(sNum*h1Num)<=conf.tol && sum(sum(abs(o_h2_mu-h2_mu),2))/(sNum*h2Num)<=conf.tol
                    converged = 1;    
%                     fprintf('optimal found %d \n',count);
                end
            end
            o_h1_mu = h1_mu;
            o_h2_mu = h2_mu;
            count = count+1;
        end
        % get reconstruction error
         X_rec = logistic(h1_mu*Ws{1}' + repmat(visB,sNum,1));
         re = re + sum(sum((X - X_rec).^2))/(sNum*visNum);        
        % get new fantasy particle
        for i=1:conf.gNum
            %h1_f = logistic(v_f*Ws{1} + h2_f*Ws{2}' + repmat(hidBs{1},M,1));
            h1_fs = 1*(h1_f > rand(size(h1_f)));
            
            h2_f = logistic(h1_fs*Ws{2} + repmat(hidBs{2},M,1));
            h2_fs = 1*(h2_f > rand(size(h2_f)));                        
            
            v_f  = logistic(h1_fs*Ws{1}' + repmat(visB,M,1));
            v_fs  = 1*(v_f > rand(size(v_f)));
            
            h1_f = logistic(v_fs*Ws{1} + h2_fs*Ws{2}' + repmat(hidBs{1},M,1));
        end        
        %update
        diff1    = (X'*h1_mu)/sNum - (v_fs'*h1_f)/M;        
        DW1      = lr*(diff1 - conf.cost*Ws{1}) + conf.mmt*DW1;        
        Ws{1}    = Ws{1} + DW1;
        
        diff2    = (h1_mu'*h2_mu/sNum - (h1_f'*h2_f)/M);
        DW2      = lr*(diff2 - conf.cost*Ws{2}) + conf.mmt*DW2;
        Ws{2}    = Ws{2} + DW2;
        
        DVB      = lr*(sum(X,1)/sNum - sum(v_fs,1)/M) + conf.mmt*DVB;
        visB     = visB  + DVB;
        
        DHB1     = lr*(sum(h1_mu,1)/sNum - sum(h1_f,1)/M) + conf.mmt*DHB1;
        hidBs{1} = hidBs{1} + DHB1;
        
        DHB2     = lr*(sum(h2_mu,1)/sNum - sum(h2_f,1)/M) + conf.mmt*DHB2;
        hidBs{2} = hidBs{2} + DHB2;
                
    end
    % Visualize reconstruction
    %if ~isempty(conf.vis_dir)
    %    save_images(X_rec,strcat(conf.vis_dir,'dbm_rec_'),sNum,e,conf.row,conf.col);
    %end
    fprintf('[Epoch %d] Reconstruction Error = %f\n',e,re/conf.bNum);
    
    %Visualize the first-level basis
    %if rem(e-1,5)==0
        MN = min(min(Ws{1}));
        MX = max(max(Ws{1}));        
        subplot(1,2,1); show_images(logistic(Ws{1}'),100,conf.row,conf.col);
        subplot(1,2,2); show_images(X_rec,sNum,conf.row,conf.col);
        drawnow;
    %end
end
end