%%%% ADDING ALL NEEDED DIRECTORY FOR PROJECTS %%%%
addpath(genpath('.'));
 sys_inf = computer();
 if ~isempty(findstr('WIN',sys_inf))
     addpath(genpath('C:\Pros\Experiments\KBDBN\DATA\'));     
 elseif ~isempty(findstr('linux',sys_inf)) || ~isempty(findstr('GLNX',sys_inf))
     addpath(genpath('/home/funzi/Documents/Experiments/KBDBN/DATA/'));
     addpath(genpath('/home/funzi/Documents/Experiments/KBDBN/DATA/'));
 end