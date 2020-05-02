function [Z] = WUDA1(Xs,Xt,options)

% 导入必要的参数
k = options.k;
lambda = options.lambda;

% 计算一些统计量
X = [Xs,Xt];
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);

% 计算MMD
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e';
M = M/norm(M,'fro');

% 中心化矩阵
H = eye(n)-1/(n)*ones(n,n);

% JDA开始
[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',k,'SM');
Z = A'*X;
end
