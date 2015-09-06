function cost = llh(outputs,labels)
   [lNum,sNum] = size(outputs);
   [~,labels] = max(labels);
   if max(labels(:)) > lNum || min(labels(:))<1
       fprintf(strcat(mfilename,': Error! Label does not match!!!'));
       cost = NaN;
   end
   inx = labels + [0:sNum-1]*lNum;
   cost = mean(log(outputs(labels)+0.0000000001));

   if isinf(cost) || isnan(cost)
       fpritnf('Error!! Cost is NaN or Inf!!\n');
   end
end