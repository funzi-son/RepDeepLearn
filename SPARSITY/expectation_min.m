% script to compute sparsity using Expectaion minimization (Lee's)
% require  - model: W, hidB, visB
%          - conf (model & training settings)
%          - input data: visP
%          - batch size: sNum
% return:  - updated model
%          - spasiry error: spr_e
% Son T - 2014

hidII = hidP;
%current sparsity
pppp = (conf.p - mean(hidII,2));
hspr = hspr + mean(pppp.^2);
sigmoid_deriv = hidII.*(1-hidII);

if isfield(conf,'sparse_w')
    w_diff = conf.lambda*(repmat(pppp',visNum,1).*(visP*sigmoid_deriv')/sNum);
else
    w_diff = 0;
end
h_diff = conf.lambda*(pppp.*(sum(sigmoid_deriv,2)/sNum));