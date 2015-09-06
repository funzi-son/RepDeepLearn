[~,host_name] = system('hostname');
lrs     = [0.001];
mms     = [0 0.001 0.01];
csts    = [0 0.001 0.01 0.05 0.1 0.5];

lds     = [0 0.01 0.1 1 10 100];
ps      = [0.0001 0.001];


host_name = strtrim(host_name);
if strcmp(host_name,'Shopbuild6038') % Uni desktop
    PRJ_DIR = 'C:\Pros\Experiments\SPARSITY\';        
    DAT_DIR = 'C:\Pros\Data\';    
    lm = '\';       
elseif strcmp(host_name,'funzi-K43SJ') % Son's laptop
    PRJ_DIR= '/home/funzi/Documents/Experiments/SPARSITY/';    
    DAT_DIR = '/home/funzi/My.Academic/DATA/CHAR.REG/';
    lm = '/';   
elseif strcmp(host_name,'funzi-X550CC') % Son's laptop
    PRJ_DIR= '/home/funzi/Documents/Experiments/SPARSITY/';    
    DAT_DIR = '/home/funzi/My.Academic/DATA/CHAR.REG/';
    lm = '/';    
elseif strcmp(host_name,'Chianti-PC') % Chi's laptop
    PRJ_DIR = 'E:\Experiments\SPARSITY\'; 
    DAT_DIR = 'E:\DATA\';    
    lm = '\';
    addpath(genpath('E:\libs\SVM\'));    
elseif strcmp(host_name,'AVELINO-XPS-8700') % AV's desktop
    PRJ_DIR= '/home/sont/SON_EXP/Experiments/SPARSITY/';        
    DAT_DIR = '/home/sont/SON_EXP/DATA/';
    lm = '/';        
elseif strcmp(host_name(1:4),'node') %SOLON CLUSTER
    PRJ_DIR= '/home/abdz481/SON_EXP/Experiments/SPARSITY/';        
    DAT_DIR = '/home/abdz481/SON_EXP/DATA/';    
    lm = '/';    
else 
    assert(1==0,'Check the experiment document setting!');
end
DAT_CASE = 1;
if DAT_CASE ==1
    EXP_DIR = strcat(PRJ_DIR,'RBM',lm,'MNIST',lm);        
    DAT_DIR = strcat(DAT_DIR,'MNIST',lm);
    
elseif DAT_CASE ==2
    EXP_DIR = strcat(PRJ_DIR,'RBM',lm,'YALE',lm);        
    DAT_DIR = strcat(strrep(DAT_DIR,'CHAR.REG','FACE'),lm,'YALE',lm);
end

SPARSITY = 'RELU';

if ~exist(EXP_DIR,'dir'), mkdir(EXP_DIR); end