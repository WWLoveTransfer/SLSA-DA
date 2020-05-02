function [Z] = WUDA1(Xs,Xt,options)

% �����Ҫ�Ĳ���
k = options.k;
lambda = options.lambda;

% ����һЩͳ����
X = [Xs,Xt];
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);

% ����MMD
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e';
M = M/norm(M,'fro');

% ���Ļ�����
H = eye(n)-1/(n)*ones(n,n);

% JDA��ʼ
[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',k,'SM');
Z = A'*X;
end
