function [acc_s,acc_t] = runTCA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,lp)

%% Set algorithm options
if lp==1
    options.lambda = 0.05;
    options.dim = 20;
else
    options.lambda = 0.1;
    options.dim = 100;
end
%% Run algorithm
[acc_s,acc_t] = TCA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);
end