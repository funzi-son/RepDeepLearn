% All inline function below treat the data D as mxn matrix
% with n data point of m dimension
if strcmp(norm,'l1') % Scaling each data to have unit L1
    norm_fnc = inline('bsxfun(@rdivide,D,sum(abs(D)))','D');
elseif strcmp(norm,'l2') % Scaling each data to have unit L2
    norm_fnc = inline('bsxfun(@rdivide,D,sqrt(sum(D.^2)))','D');
elseif strcmp(norm,'minmax') % Scaling each feature to MN and MX    
    norm_fnc = inline('refine_data(bsxfun(@rdivide,bsxfun(@minus,D,min(D,[],2)),max(D,[],2)-min(D,[],2))*(MX-MN) + MN)','D','MN','MX');
elseif strcmp(norm,'mean-var')
    norm_fnc = inline('refine_data(bsxfun(@rdivide,bsxfun(@minus,D,mean(D,2)),std(D,0,2)))*V+M','D','M','V');
end


