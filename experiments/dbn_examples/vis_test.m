function  vis_test()
EXP_DIR = '~/Documents/Experiments/RULE_EXT/IMG/MNIST/DBN_RULE/20k/MODEL_EVAL/';
fs = dir(strcat(EXP_DIR,'log*'));
for i=1:size(fs,1)
    model = load_deepnet(EXP_DIR,fs(i).name);
    model
    pause
    figure(1);visualize_1l_filters(model(1).W,100,28,28,'minmax',1);
    figure(2);visualize(model,100,28,28,'minmax');
    pause
end

end

