function err = logreg_mse(outputs,labels)
   diff = (labels - outputs);
   expdiff = sum(diff.*outputs);
   err= bsxfun(@minus,diff,expdiff).*outputs;
end
