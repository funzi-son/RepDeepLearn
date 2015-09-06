if strcmp(obj_fnc,'mle')
    error_fnc = @logreg_mle;
elseif strcmp(obj_fnc,'mse')
    error_fnc = @logreg_mse;
elseif strcmp(obj_fnc,'cent')
    error_fnc = @logreg_cent;
else
    fprintf('Error!! No objective function is set!!!\n');
end