function [Cls] = GraphClassifier3(Xs,Xt,Ys,delta_s,Cs,Asu,kk)

% 导入数据
X = [Xs,Xt];
ns = size(Xs,2);
nt = size(Xt,2);
n = ns+nt;
C = length(unique(Ys));

% 求得标签矩阵，只有源领域，目标领域全为0
T = zeros(n,C);
for ik=1:size(Ys)
    T(ik,Ys(ik))=1;
end

% 初始化一些参数
k = kk; % 关于几近邻的参数，字符数据集默认是10

% 关于标签传播的3个参数
is_normalize = 1;

% 为了简单起见，这里用欧式距离
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2); % 对其进行排序

% 初始化相似矩阵和权重
W = zeros(n);

% 求相似矩阵
for i = 1:n
    di = distX1(i,2:k+2); % 找出离除自己外最近的k+1个点
    
    % 求权重 源域到自己的权重大，到目标域小，防止目标域全0
    id = idx(i,2:k+2); % 并且找出对应的索引
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); % 最后一个的权值为0，越小越大，因此只定义了最k近邻的权重
end;

% 为了使得W对称
W0 = (W+W')/2;

% 开始分类
F = GeneralSSL3(W0, T, is_normalize, delta_s, Cs, Asu);
F = normr(F);
[~,Cls]=max(F,[],2);
end

