function [Z,A] = WUDA2(Xs,Xt,Ys,Yt0,options,MC)

% �����Ҫ�Ĳ���
k = options.k;
lambda = options.lambda;
gamma =options.gamma;
% ����һЩͳ����
X = [Xs,Xt];
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));
class = unique(Ys);

% ����Դ��LDA����
dim = size(Xs,1);
meanTotal = mean(Xs,2);
Sws = zeros(dim, dim);
Sbs = zeros(dim, dim);

for i=1:C
    Xi = Xs(:,find(Ys==class(i)));
    meanClass = mean(Xi,2);
    Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
    Sws = Sws + Xi*Hi*Xi'; 
    Sbs = Sbs + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
end

% ����Ŀ����LDA����
Swt = zeros(dim, dim);
Sbt = zeros(dim, dim);

if ~isempty(Yt0) && length(Yt0)==nt
for i=1:C
    Xi = Xt(:,find(Yt0==class(i)));
    meanClass = mean(Xi,2);
    Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
    Swt = Swt + Xi*Hi*Xi'; 
    Sbt = Sbt + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
end
end

% ����MMD
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C;
M=M+MC;
M = M/norm(M,'fro');

% ���Ļ�����
H = eye(n)-1/(n)*ones(n,n);

% JDA��ʼ
[A,~] = eigs(X*M*X'+lambda*eye(m)+gamma*(Sws)+gamma*(Swt),X*H*X',k,'SM');
Z = A'*X;
end
