function [acc_s,acc_t] = runVDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,lp)

%% Set algorithm options
if lp==1
    options.lambda = 0.05;
    options.dim = 20;
else
    options.lambda = 0.1;
    options.dim = 100;
end

options.T = 5;
options.gamma = 0.01;  % 类内类间
%% Run algorithm
[acc_s,acc_t] = VDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);
end