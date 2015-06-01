function [H,W] = nmf(conf,X)
% Non-negative matrix factorization: Lee & Seung NIPS 2001
% with sparse contraints from Hoyer JMLR-2004
% and sparse normalization from Liu et al ICASSP-2003
% Son T-2015
% Note that from now-on all data matrix will be seen as: nxm where m is the
% number of samples, change the old code accordingly
factorizing = 1;
[visNum,sz] = size(X);
hidNum = conf.hidNum;

W = abs(randn(visNum,hidNum));
H = abs(randn(hidNum,sz));
cost_func = inline('sum(sum((X - WH*).^2))','X','W','H'); % Euclidean cost function -  default

w_norm = 0;
h_norm = 0;
alpha = 0;
if isfield(conf,'w_norm'), w_norm = conf.w_norm; end
if isfield(conf,'h_norm'), h_norm = conf.h_norm; end
epoch = 1;
err = [];
proj_sparsity=[0,0];

if isfield(conf,'h_proj_l1_norm'), proj_sparsity(1) = (conf.h_proj_l1_norm ~=0); end
if isfield(conf,'w_proj_l1_norm'), proj_sparsity(2) = (conf.w_proj_l1_norm ~=0); end    
if isfield(conf,'sparse_alpha'), conf.costFnc = 'reg i-divergence'; alpha = conf.sparse_alpha; end
    
if sum(proj_sparsity)>0 % sparsity with projection (Hoyer)
    conf.costFnc = 'euclidean';
    h_norm = 1;
    if proj_sparsity(1)
        h_step = 1;
        % normalize H to have unit L2norm
        H = bsxfun(@rdivide,H,sqrt(sum(H.^2,2)));
        % apply sparseness        
        h_l1_init = sqrt(sz) -(sqrt(sz)-1)*conf.h_proj_l1_norm;
        for i=1:size(H,1)
            H(i,:) = projfunc(H(i,:)',h_l1_init,1,1)';
        end
    end
    if proj_sparsity(2)
        w_step = 1;        
        % apply sparseness
        w_l1_init = sqrt(visNum) - (sqrt(visNum)-1)*conf.w_proj_l1_norm;
        for i=1:size(W,2)
            W(:,i) = projfunc(W(:,i),w_l1_init,1,1);
        end
        
    end
end

if strcmp(conf.costFnc,'euclidean')    
    w_update = inline('W.*((X*transpose(H))./max((W*H)*transpose(H),1e-20))','X','W','H');
    h_update = inline('H.*((transpose(W)*X)./max(transpose(W)*(W*H),1e-20))','X','W','H');
elseif strcmp(conf.costFnc,'i-divergence')    
    cost_func = inline('sum(sum(X.*log(max(X,1e-20)./max(W*H,1e-20)) - X + W*H))','X','W','H');
    w_update = inline('W.*bsxfun(@rdivide,(X./max((W*H),1e-20))*transpose(H),transpose(sum(H,2)))','X','W','H');
    h_update = inline('H.*bsxfun(@rdivide,transpose(W)*(X./max((W*H),1e-20)),transpose(sum(W)))','X','W','H');
elseif strcmp(conf.costFnc,'reg i-divergence')
    cost_func = inline('sum(sum(X.*log(max(X,1e-20)./max(W*H,1e-20)) - X + W*H)) + alpha*sum(sum(H))','X','W','H','alpha');
    w_update = inline('W.*bsxfun(@rdivide,(X./max((W*H),1e-20))*transpose(H),transpose(sum(H,2)))','X','W','H');
    h_update = inline('H.*(transpose(W)*(X./max(W*H,1e-20)))/(1+alpha)','X','W','H','alpha');
    w_norm = 1;
end

while factorizing,        
        % Update H                                   
        if proj_sparsity(1)
            obj_reduce = 0;
            h_grad = W'*(X-W*H);
            current_obj_cost = sum(sum((X-W*H).^2));
            while ~obj_reduce % while objective cost is not reduced
                H_ = H + h_step*h_grad;
                for i=1:size(H_,1)
                    H_(i,:) = projfunc(H_(i,:)',h_l1_init,1,1)';
                end
                obj_cost = sum(sum((X-W*H_).^2));
                if obj_cost<=current_obj_cost, obj_reduce = 1; 
                else 
                    h_step = h_step/2;
                    if h_step<1e-200, return; end
                end                
            end
            h_step = h_step*1.2;
            H = H_;
            
        else
            H = h_update(X,W,H,alpha); 
            
            if h_norm % L2 norm
            % Normalize H such that l2_h = 1;
                l2h = sqrt(sum(H.^2,2));
                H = bsxfun(@rdivide,H,l2h);
                if sum(proj_sparsity)>0 % Hoyer did this
                    W = bsxfun(@times,W,l2h');
                end
            end
        end
        
        % Update W                
        if proj_sparsity(2) % Doing projection of 
            obj_reduce =0;
            w_grad = (X-W*H)*H';
            current_obj_cost = sum(sum((X-W*H).^2));
            while ~obj_reduce % while the objective cost is not reduced
                W_ = W + w_step*w_grad; % This indeed decreases the objtive cost but the one below may not
                w_l2 = sum(W_.^2);
                w_l1 = w_l1_init*sqrt(w_l2);
                for i=1:size(W_,2), W_(:,i) = projfunc(W_(:,i),w_l1(i),w_l2(i),1); end
                obj_cost = sum(sum((X-W_*H).^2));
                %disp([current_obj_cost,obj_cost]);
                if obj_cost<=current_obj_cost, obj_reduce = 1;
                else
                    fprintf('.');
                    w_step = w_step/2; % If the obj function is not reduced, decrease step and try one more                    
                    if w_step<1e-200, return; end
                end                                
            end
            fprintf('\n');
            w_step = w_step*1.2; 
            W = W_;
        else 
            W = w_update(X,W,H);            
            if w_norm, W = bsxfun(@rdivide,W,sum(W)); end % L1 norm
        end
        
        %% FOR DEMO & DEBUG ONLY, COMMENT WHEN RUNNING FOR BETTER PERFORMANCE
        visualize_1l_filters(W,100,conf.row,conf.col,'minmax',conf.row_order);
        pause(0.001);
        e = cost_func(X,W,H,alpha);
        err = [err e];
        fprintf('[Epoch %3d] Cost = %.5f \n',epoch,e);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        epoch = epoch+1;
        if epoch>conf.eNum, factorizing=0; end          
end
figure();
plot(err);
end

function [v,usediters] = projfunc( s, k1, k2, nn )

% Solves the following problem:
% Given a vector s, find the vector v having sum(abs(v))=k1 
% and sum(v.^2)=k2 which is closest to s in the euclidian sense.
% If the binary flag nn is set, the vector v is additionally
% restricted to being non-negative (v>=0).
%    
% Written 2.7.2004 by Patrik O. Hoyer
%

% Problem dimension
N = length(s);

% If non-negativity flag not set, record signs and take abs
if ~nn,
    isneg = s<0;
    s = abs(s);
end

% Start by projecting the point to the sum constraint hyperplane
v = s + (k1-sum(s))/N;

% Initialize zerocoeff (initially, no elements are assumed zero)
zerocoeff = [];

j = 0;
while 1,

    % This does the proposed projection operator
    midpoint = ones(N,1)*k1/(N-length(zerocoeff));
    midpoint(zerocoeff) = 0;
    w = v-midpoint;
    a = sum(w.^2);
    b = 2*w'*v;
    c = sum(v.^2)-k2;
    alphap = (-b+real(sqrt(b^2-4*a*c)))/(2*a);
    v = alphap*w + v;
    
    if all(v>=0),
	% We've found our solution
	usediters = j+1;
	break;
    end
        
    j = j+1;
        
    % Set negs to zero, subtract appropriate amount from rest
    zerocoeff = find(v<=0);
    v(zerocoeff) = 0;
    tempsum = sum(v);
    v = v + (k1-tempsum)/(N-length(zerocoeff));
    v(zerocoeff) = 0;
            
end

% If non-negativity flag not set, return signs to solution
if ~nn,
    v = (-2*isneg + 1).*v;
end

% Check for problems
if max(max(abs(imag(v))))>1e-10,
    error('Somehow got imaginary values!');
end
end