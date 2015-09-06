% script to compute sparsity using Cross Entropy (Andrew's)
% require  - model: W, hidB, visB
%          - conf (model & training settings)
%          - input data: visP
%          - batch size: sNum
% return:  - updated model
%          - spasiry error: spr_e
% Son T - 2014

hidII = hidP;

if isfield(conf,'cumsparse') && conf.cumsparse
%current sparsity
    if ~exist('phat','var')
        phat = mean(hidP,2);
    else
        q_prev = phat;
        phat = 0.9*q_prev+0.1*mean(hidP,2);
    end     
else
    phat = mean(hidII,2);
end
hspr = hspr + mean(conf.p*log(conf.p./phat) + (1-conf.p)*log((1-conf.p)./(1-phat)));

sigmoid_deriv = hidII.*(1-hidII);
phatpp = (conf.p-phat)./(phat.*(1-phat));
w_diff = conf.lambda*(bsxfun(@times,(visP*sigmoid_deriv')/sNum,phatpp'));
h_diff = conf.lambda*(phatpp.*(sum(sigmoid_deriv,2)/sNum));