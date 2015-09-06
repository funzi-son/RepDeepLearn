function vis_exp_dbn( )
% THIS ONE USE OLD FORMAT DATA
load('~/My.Academic/DATA/CHAR.REG/MNIST/mnist_train_dat_10k.mat');
rbm = load('~/Documents/Experiments/A_REPTRAN/vis_exp/mnist_rbm_sparse2.mat');
model(1).W = rbm.model.W; model(1).U = [];
model(1).visB = rbm.model.visB'; model(1).hidB = rbm.model.hidB';
 model(1).labB = [];
clear rbm;
traind = traind';
traind = logistic(bsxfun(@plus,model(1).W'*traind,model(1).hidB));
trn_fts_file = '~/Documents/Experiments/tmp/rbm1_output.mat';
save(trn_fts_file,'traind');

normalize = 'single_minmax';
visualize_1l_filters(model(1).W,100,28,28,normalize,1,'~/Documents/Experiments/tmp/fist_filter.bmp');

DBN3 = 1;
for lr = [0.5]
for mm = [0.01]
for cst = [0.001]
for ld = [0]% 0.01 0.1 1 5 10]
for p = [0.0001]
    img_name = strcat('~/Documents/Experiments/tmp/lr_',num2str(lr),...
        '_mm',num2str(mm),'_cst',num2str(cst),'_ld',num2str(ld),'_p',num2str(p),'.bmp');
    mod_file = strrep(img_name,'.bmp','dbn2.mat');
    out_file = strrep(img_name,'.bmp','out.mat');
    if ~exist(mod_file,'file')
        conf.hidNum  = 500;
        conf.eNum = 100;
        conf.bNum = 100;
        conf.sNum = 100;
        conf.gNum = 1;    
        conf.params = [lr lr mm cst];
        conf.sparsity  = 'EMIN';% EMIN,KLMIN    
        conf.sparse_w  = 1; % Only for EMIN, apply sparsity to w or not(for Lee's approach)
        conf.lambda = ld;
        conf.p      = p;

        conf.trn_dat_file  = trn_fts_file;           
        model(2) = gen_rbm_train(conf);   
        save(mod_file,'model');                
    else        
        load(mod_file,'model');
    end    
    visualize(model,100,28,28,normalize,img_name);
    if DBN3
        outdat = logistic(bsxfun(@plus,model(2).W'*traind,model(2).hidB));
        save(out_file,'outdat');
        for lr2 = [0.01]
        for cst = [0.001]        
             conf1.hidNum  = 1000;
                conf1.eNum = 200;
                conf1.bNum = 100;
                conf1.sNum = 100;
                conf1.gNum = 1;    
                conf1.params = [lr2 lr2 0.01 cst];
                conf1.sparsity  = 'EMIN';% EMIN,KLMIN    
                conf1.sparse_w  = 1; % Only for EMIN, apply sparsity to w or not(for Lee's approach)
                conf1.lambda = 1;
                conf1.p      = 0.0001;
                conf1.trn_dat_file  = out_file;    
            final_mod_file = strcat('~/Documents/Experiments/tmp/final',num2str(lr2),'_',num2str(cst),...
                '_',num2str(conf1.lambda),'_',num2str(conf1.p),'.mat');        
            if ~exist(final_mod_file,'file')                      
                model(3) = gen_rbm_train(conf1);
                save(final_mod_file,'model');
            else
                load(final_mod_file);                           
            end
            visualize(model,100,28,28,normalize,...
            strcat('~/Documents/Experiments/tmp/final',num2str(lr2),'_',num2str(cst),...
                '_',num2str(conf1.lambda),'_',num2str(conf1.p),'.bmp'));
        end
        end
    end
end
end
end
end
end
end
