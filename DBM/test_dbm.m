function test_dbm()
sys_inf = computer();
if ~isempty(findstr('WIN',sys_inf))
    EXP_DIR = 'C:/Pros/Experiments/DIS_STF';
    lm = '\'
elseif ~isempty(findstr('linux',sys_inf)) || ~isempty(findstr('GLNX',sys_inf))     
    EXP_DIR = '/home/funzi/Documents/Experiments/DIS_STF/';
    lm = '/';
end
%test_dbm
    mnist_trn_dat_file = 'mnist_train_dat_10k.mat';
    mnist_tst_dat_file = 'mnist_test_dat_10k.mat';
    
    vars = whos('-file', mnist_trn_dat_file);
    A = load(mnist_trn_dat_file,vars(1).name);
    trn_features = A.(vars(1).name);
        
    vars = whos('-file', mnist_tst_dat_file);
    A = load(mnist_tst_dat_file,vars(1).name);
    tst_features = A.(vars(1).name);
    
    clear A;
    H = 28;
    W = 28;
    
    
    dbm_conf.sNum = 100;
    dbm_conf.bNum = 5;
    dbm_conf.model  = strcat(EXP_DIR,'TEST',lm,'dbm');
    dbm_conf.vis_dir = strcat(EXP_DIR,'TEST',lm,'VIS',lm);
    
    dbm_conf.layer(1).hidNum = 500;
    dbm_conf.layer(1).eNum   = 100;
    dbm_conf.layer(1).bNum   = dbm_conf.bNum;
    dbm_conf.layer(1).sNum   = dbm_conf.sNum;
    dbm_conf.layer(1).gNum   = 1;
    dbm_conf.layer(1).params = [0.5 0.2 0.1 0.00002];   
    dbm_conf.layer(1).N = 50;
    dbm_conf.layer(1).MAX_INC = 10;
    dbm_conf.layer(1).plot_ = 0;        
    dbm_conf.layer(1).vis_dir  = dbm_conf.vis_dir;
    dbm_conf.layer(1).row     = H;
    dbm_conf.layer(1).col     = W;
    
    dbm_conf.layer(2).hidNum = 1000;
    dbm_conf.layer(2).eNum   = 100;
    dbm_conf.layer(2).bNum   = dbm_conf.bNum;
    dbm_conf.layer(2).sNum   = dbm_conf.sNum;
    dbm_conf.layer(2).gNum   = 1;
    dbm_conf.layer(2).params = [0.5 0.2 0.1 0.00002];  
    dbm_conf.layer(2).N = 50;
    dbm_conf.layer(2).MAX_INC = 10;
    dbm_conf.layer(2).plot_ = 0;            
        
    dbm_conf.gen.hidNum1 = dbm_conf.layer(1).hidNum;
    dbm_conf.gen.hidNum2 = dbm_conf.layer(2).hidNum;
    dbm_conf.gen.eNum    = 100;
    dbm_conf.gen.bNum    = dbm_conf.bNum;
    dbm_conf.gen.sNum    = dbm_conf.sNum;
    dbm_conf.gen.gNum    = 10;
    dbm_conf.gen.lr      = 0.5;
    dbm_conf.gen.mmt     = 0.1;
    dbm_conf.gen.cost    = 0.00002;
    
    dbm_conf.gen.M        = 100;
    dbm_conf.gen.max_iter = 10;
    dbm_conf.gen.tol      = 0.00000001;
    
    dbm_conf.gen.row     = H;
    dbm_conf.gen.col     = W;
    dbm_conf.gen.vis_dir  = dbm_conf.vis_dir;
    
    %% Training
    trn_features = trn_features(1:dbm_conf.sNum*dbm_conf.bNum,:);
    [Ws visBs hidBs] = training_dbm_(dbm_conf,trn_features);
    %% Sampling
    load(strcat(dbm_conf.model,'.mat'));
    h = plot(nan);
    v_f  = round(rand(dbm_conf.gen.M,dbm_conf.gen.row*dbm_conf.gen.col));
    h1_f = logistic(v_f*(2*Ws{1}) + repmat(hidBs{1},dbm_conf.gen.M,1));    
    for i=1:10000        
        h1_fs = 1*(h1_f > rand(size(h1_f)));
            
        h2_f = logistic(h1_fs*Ws{2} + repmat(hidBs{2},dbm_conf.gen.M,1));
        h2_fs = 1*(h2_f > rand(size(h2_f)));                        
            
        v_f  = logistic(h1_fs*Ws{1}' + repmat(visB,dbm_conf.gen.M,1));       
        v_fs  = 1*(v_f > rand(size(v_f)));
        show_images(v_f,dbm_conf.gen.M,dbm_conf.gen.row,dbm_conf.gen.col);            
        drawnow;
    	h1_f = logistic(v_fs*Ws{1} + h2_fs*Ws{2}' + repmat(hidBs{1},dbm_conf.gen.M,1));
    end
end

