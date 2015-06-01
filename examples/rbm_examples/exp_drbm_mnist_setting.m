[~,host_name] = system('hostname');
host_name = strtrim(host_name);
lrs     = [0.005];
mms     = [0.001 0.01 0.1 0.5];
csts    = [0.00001 0.0001];


if strcmp(host_name,'Shopbuild6038') % Uni desktop
    PRJ_DIR = 'C:\Pros\Experiments\RULE_EXT\';
    DAT_DIR = 'C:\Pros\Data\';
    lm = '\';      
elseif strcmp(host_name,'funzi-K43SJ') % Son's laptop
    PRJ_DIR= '/home/funzi/Documents/Experiments/RULE_EXT/';    
    DAT_DIR = '/home/funzi/My.Academic/DATA/CHAR.REG/';
    lm = '/'; 
elseif strcmp(host_name,'funzi-X550CC') % Son's laptop
    PRJ_DIR= '/home/funzi/Documents/Experiments/RULE_EXT/';    
    DAT_DIR = '/home/funzi/My.Academic/DATA/CHAR.REG/';
    lm = '/';     
elseif strcmp(host_name,'Chianti-PC') % Chi's laptop
    PRJ_DIR = 'E:\Experiments\RULE_EXT\'; 
    DAT_DIR = 'E:\DATA\';    
    lm = '\';    
elseif strcmp(host_name,'AVELINO-XPS-8700') % AV's desktop
    PRJ_DIR= '/home/sont/SON_EXP/Experiments/RULE_EXT/';
    DAT_DIR = '/home/sont/SON_EXP/DATA/';
    lm = '/';
elseif strcmp(host_name(1:4),'node') %SOLON CLUSTER
    PRJ_DIR= '/home/abdz481/SON_EXP/Experiments/RULE_EXT/';        
    DAT_DIR = '/home/abdz481/SON_EXP/DATA/';
    lm = '/';
else
    assert(1==0,'Check the experiment document setting!');
end


EXP_DIR = strcat(PRJ_DIR,'IMG',lm,'MNIST',lm,'RBM',lm,'60k',lm);    
EXP_MOD = strcat(PRJ_DIR,'IMG',lm,'MNIST',lm,'RBM',lm,'60k',lm);
if ~exist(EXP_DIR,'dir'), mkdir(EXP_DIR); end
if ~exist(EXP_MOD,'dir'), mkdir(EXP_MOD); end
DAT_DIR  = strcat(DAT_DIR,'MNIST',lm);