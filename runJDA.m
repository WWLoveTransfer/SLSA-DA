function [acc_s,acc_t] = runJDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,lp)

%% Set algorithm options
if lp==1
    options.lambda = 0.05;
    options.dim = 20;
else
    options.lambda = 0.1;
    options.dim = 100;
end
options.T = 5;
%% Run algorithm
[acc_s,acc_t] = JDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);
end