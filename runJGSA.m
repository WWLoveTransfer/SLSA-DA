function [acc_s,acc_t] = runJGSA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,lp)

% Set algorithm parameters
if lp==1
    options.k = 20; 
else
    options.k = 100;
end           % subspace base dimension
options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly

options.T = 5;             % #iterations, default=10

options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.1;         % the parameter for P and Q (source discriminaiton)
options.gamma = 2;          % the parameter for kernel

Xs = [Xs_l;Xs_u];
ns_l = size(Xs_l,1);
Xs = normr(Xs)';
Xt = normr(Xt)';
Xs_l = Xs(:,1:ns_l);
Xs_u = Xs(:,ns_l+1:end);
[acc_s,acc_t] = JGSA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);
end