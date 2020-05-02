function [acc_s,acc_t] = runMEDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,lp)
if lp==1
    options.d = 20;
else
    options.d = 100;
end
options.rho = 1.0;
options.p = 10;
options.lambda = 10.0;
options.eta = 0.1;
options.T = 5;
[acc_s,acc_t] = MEDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);

end

