function err = logreg_cent(outputs,labels)
   diff = (labels-outputs)./(outputs.*(1-outputs)+0.0000000001);
   expdiff = sum(diff.*outputs);
   err= bsxfun(@minus,diff,expdiff).*outputs;
end