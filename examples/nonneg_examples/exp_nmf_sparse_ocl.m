function exp_nmf_sparse_ocl()

V = get_data_from_file('orl_face.mat');
MN = min(V(:));
MX = max(V(:));
V = (V-MN)/(MX-MN);
show_images(V',100,56,46,0);

pause
conf.hidNum = 100;
conf.eNum = 10000;
conf.stop = 0.001;
conf.eps = 0.0000001;
conf.costFnc = 'euclidean';

%% NMF with sparsity constraints
%conf.sparsity = [0,1]; % Sparsity for H, W
%conf.w_proj_l1_norm = 0.75;
%conf.h_proj_l1_norm = 0;

%% Sparse NMF
conf.sparse_alpha = 5;

conf.row = 56;
conf.col = 46;
conf.row_order = 0;

nmf(conf,V);
end

