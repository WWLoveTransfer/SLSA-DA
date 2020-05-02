function [Cls] = GraphClassifier3(Xs,Xt,Ys,delta_s,Cs,Asu,kk)

% ��������
X = [Xs,Xt];
ns = size(Xs,2);
nt = size(Xt,2);
n = ns+nt;
C = length(unique(Ys));

% ��ñ�ǩ����ֻ��Դ����Ŀ������ȫΪ0
T = zeros(n,C);
for ik=1:size(Ys)
    T(ik,Ys(ik))=1;
end

% ��ʼ��һЩ����
k = kk; % ���ڼ����ڵĲ������ַ����ݼ�Ĭ����10

% ���ڱ�ǩ������3������
is_normalize = 1;

% Ϊ�˼������������ŷʽ����
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2); % �����������

% ��ʼ�����ƾ����Ȩ��
W = zeros(n);

% �����ƾ���
for i = 1:n
    di = distX1(i,2:k+2); % �ҳ�����Լ��������k+1����
    
    % ��Ȩ�� Դ���Լ���Ȩ�ش󣬵�Ŀ����С����ֹĿ����ȫ0
    id = idx(i,2:k+2); % �����ҳ���Ӧ������
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); % ���һ����ȨֵΪ0��ԽСԽ�����ֻ��������k���ڵ�Ȩ��
end;

% Ϊ��ʹ��W�Գ�
W0 = (W+W')/2;

% ��ʼ����
F = GeneralSSL3(W0, T, is_normalize, delta_s, Cs, Asu);
F = normr(F);
[~,Cls]=max(F,[],2);
end

