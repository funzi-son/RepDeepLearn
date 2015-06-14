function exp_nn()
close all;
clc;
clear;
conf.hidNum = [20 40];
conf.activationFnc = {'tansig','tansig','logsig'}; % tansig, logsig, purelin
conf.eNum       = 100;
conf.bNum       = 0; % set later in each dataset
conf.sNum       = 0; % set later in each dataset
conf.params(1) = 0.5;            % learning rate
conf.params(2) = conf.params(1); % not used
conf.params(3) = 0.1;            % momentum
conf.params(4) = 0.0001;         % cost

conf.E_STOP_LR_REDUCE = 50;      % Number of bad updates before reduce the learning rate
conf.E_STOP = 5;                 % Number of learning rate decay before stop training;

CASE = 2;

if CASE==1
    load glass_dataset;
    all_dat = glassInputs';
    all_dat = bsxfun(@rdivide,bsxfun(@minus,all_dat,min(all_dat)),(max(all_dat)-min(all_dat)));
    [~,all_lab] = max(glassTargets',[],2); all_lab = all_lab-1;
    trn_dat = [];trn_lab = [];vld_dat = [];vld_lab=[];
    for l=unique(all_lab)'
        inx= find(all_lab==l);
        trn_num = round(0.7*size(inx,1)); % get 60% of this label for training
        trn_dat = [trn_dat;all_dat(inx(1:trn_num),:)];        
        vld_dat = [vld_dat;all_dat(inx(trn_num+1:end),:)];        
        
        trn_lab = [trn_lab;all_lab(inx(1:trn_num))];
        vld_lab = [vld_lab;all_lab(inx(trn_num+1:end))];
    end
    trn_dat = trn_dat';
    
    vld_dat = vld_dat';    
    conf.bNum = 1; conf.sNum = size(trn_dat,2);    
elseif CASE==2
    conf.bNum = 50;conf.sNum = 100;    
    trn_dat_file = 'mnist_train_dat_20k.mat';
    trn_lab_file = 'mnist_train_lab_20k.mat';
    vld_dat_file = 'mnist_vld_dat_10k.mat';
    vld_lab_file = 'mnist_vld_lab_10k.mat';
    tst_dat_file = 'mnist_test_dat_10k.mat';
    tst_lab_file = 'mnist_test_lab_10k.mat';
    
    trn_dat = get_data_from_file(trn_dat_file,1);
    trn_lab = get_data_from_file(trn_lab_file)';
    vld_dat = get_data_from_file(vld_dat_file,1);
    vld_lab = get_data_from_file(vld_lab_file)';
    tst_dat = get_data_from_file(tst_dat_file,1);
    tst_lab = get_data_from_file(tst_lab_file)';
end


model = train_nn(conf,[],[],trn_dat,trn_lab,vld_dat,vld_lab);

%% 
if CASE==2, visualize_1l_filters(model.Ws{1},10,28,28,'minmax'); end

if exist('tst_dat','var')
   cout = run_nn(conf.activationFnc,model,tst_dat);
   tst_acc = sum((cout-1)==tst_lab)/size(tst_lab,1);
   fprintf('Test accuracy = %.5f\n',tst_acc);
end

end

