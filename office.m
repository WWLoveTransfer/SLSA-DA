clc;
close all;
clear all;

%% data loader
folder_now = pwd; addpath([folder_now, '\data']);

%% function loader
folder_now = pwd; addpath([folder_now, '\func']);

%% parameters setting
options.k = 20;  % dimensionality
options.lambda = 0.05;  % scale regularization
options.gamma = 0.01;  % projected clustering
T = 5;  % iteration
lp=1;
%% DA tasks
srcStr = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};
RES=[]; 

for iData = 1:12
wrg=0;
IIDD=[];

accTCA_s=[]; accTCA_t=[];
accJDA_s=[]; accJDA_t=[];
accBDA_s=[]; accBDA_t=[];
accVDA_s=[]; accVDA_t=[];
accJGSA_s=[]; accJGSA_t=[];
accMEDA_s=[]; accMEDA_t=[];
accFSDA_s=[]; accFSDA_t=[];

for iter=1:1000

iter
src = char(srcStr{iData});
tgt = char(tgtStr{iData});
data = strcat(src,'_vs_',tgt);

%% source domain
load(['data/' src '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);
Xs = Xs';
Ys = labels;

%% target domain
load(['data/' tgt '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);
Xt = Xt';
Yt = labels;

%% some statistical information
m=size(Xs,1);
ns=size(Xs,2);
nt=size(Xt,2);
n=ns+nt;
C=length(unique(Ys));

%% random selection

num=5; % source labeled number
[Xs_l,Xs_u,Ys_l,Ys_u] = select_labeled(Xs, Ys, num);

%% run TCA
[acc_s,acc_t] = runTCA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('TCA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accTCA_s(1,iter)=acc_s;
accTCA_t(1,iter)=acc_t;

%% run JDA
[acc_s,acc_t] = runJDA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('JDA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accJDA_s(1,iter)=acc_s;
accJDA_t(1,iter)=acc_t;

%% run BDA
[acc_s,acc_t] = runBDA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('BDA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accBDA_s(1,iter)=acc_s;
accBDA_t(1,iter)=acc_t;

%% run VDA
[acc_s,acc_t] = runVDA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('VDA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accVDA_s(1,iter)=acc_s;
accVDA_t(1,iter)=acc_t;

%% run JGSA
[acc_s,acc_t] = runJGSA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('JGSA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accJGSA_s(1,iter)=acc_s;
accJGSA_t(1,iter)=acc_t;

%% run MEDA
[acc_s,acc_t] = runMEDA(Xs_l',Xs_u',Xt',Ys_l,Ys_u,Yt,lp);
acc_s=acc_s*100;
acc_t=acc_t*100;
fprintf('MEDA: \n');
fprintf('Grap_source+acc_s=%0.1f\n',full(acc_s));
fprintf('Grap_source+acc_t=%0.1f\n',full(acc_t));
accMEDA_s(1,iter)=acc_s;
accMEDA_t(1,iter)=acc_t;

%% data normalization
X = [Xs_l,Xs_u,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
Xs_l = X(:,1:size(Xs_l,2));
Xs_u = X(:,size(Xs_l,2)+1:ns);
Xs=[Xs_l,Xs_u];
Xt = X(:,ns+1:end);
ns_l=size(Xs_l,2);

%% initialization
fprintf('FSDA:  data=%s\n',data);
%% initialize projection
[Z] = WUDA1(Xs,Xt,options);
Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
Zs = Z(:,1:size(Xs,2));
Zt = Z(:,size(Xs,2)+1:end);
Zs_l=Zs(:,1:size(Xs_l,2));
Zs_u=Zs(:,size(Xs_l,2)+1:ns);

%% initialize source labels
kk=10;
[Cls] = GraphClassifier(Zs_l,Zs_u,Ys_l,kk);
Cls_su=Cls;
Ys0=[Ys_l;Cls_su];
acc_su = length(find(Ys0==[Ys_l;Ys_u]))/length(Ys0)*100;

%% initialize target labels
kk=20;
[Cls] = GraphClassifier([Zs_l,Zs_u],Zt,Ys0,kk);
Cls_tu=Cls;
acc_tu = length(find(Cls_tu==Yt))/length(Yt)*100;
Yt0 = Cls_tu;

%% initialize cluster centriods
Fs=LabelFormat(Ys0);
Gs=Fs/(Fs'*Fs);
Ft=LabelFormat(Yt0);
Gt=Ft/(Ft'*Ft);

try
    MC=[Gs*Gs',-Gs*Gt';
            -Gt*Gs',Gt*Gt'];
    IIDD=[IIDD,iter];    
catch
    wrg=wrg+1;
    accTCA_s(1,iter)=0;
    accTCA_t(1,iter)=0;
    accJDA_s(1,iter)=0;
    accJDA_t(1,iter)=0;
    accBDA_s(1,iter)=0;
    accBDA_t(1,iter)=0;
    accVDA_s(1,iter)=0;
    accVDA_t(1,iter)=0;
    accJGSA_s(1,iter)=0;
    accJGSA_t(1,iter)=0;
    accMEDA_s(1,iter)=0;
    accMEDA_t(1,iter)=0;
    accFSDA_s(1,iter)=0;
    accFSDA_t(1,iter)=0;
    continue;
end
MC(isinf(MC)) = 0; 
%% Start
for t = 1:T 
    
    %% update projection   
    [Z,A] = WUDA2(Xs,Xt,Ys0,Yt0,options,MC);
     Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
     Zs = Z(:,1:size(Xs,2));
     Zt = Z(:,size(Xs,2)+1:end);
     Zs_l=Zs(:,1:ns_l);
     Zs_u=Zs(:,ns_l+1:end);
     
    %% update cluster centriods
     mu1=0.01; mu2=0.01;
     Fs=Fs';
     Ft=Ft';
     As=A'*Xs;
     At=A'*Xt;
    
     [Gs] = SolveG1(As, At, Gs, Gt, Fs, mu1);
     [Gt] = SolveG2(As, At, Gs, Gt, Ft, mu2);
     
    try
        MC=[Gs*Gs',-Gs*Gt';
                -Gt*Gs',Gt*Gt'];
    catch
        wrg=wrg+1;
        accTCA_s(1,iter)=0;
        accTCA_t(1,iter)=0;
        accJDA_s(1,iter)=0;
        accJDA_t(1,iter)=0;
        accBDA_s(1,iter)=0;
        accBDA_t(1,iter)=0;
        accVDA_s(1,iter)=0;
        accVDA_t(1,iter)=0;
        accJGSA_s(1,iter)=0;
        accJGSA_t(1,iter)=0;
        accMEDA_s(1,iter)=0;
        accMEDA_t(1,iter)=0;
        accFSDA_s(1,iter)=0;
        accFSDA_t(1,iter)=0;
        break;
    end

    MC(isinf(MC)) = 0;
     
    %% update source soft labels
     delta_s=0.01;
     Cs=As*Gs;
     Asu=As(:,ns_l+1:end);
     kk=10;
     [Cls] = GraphClassifier3(Zs_l,Zs_u,Ys_l,delta_s,Cs,Asu,kk);
     Cls_su=Cls;
     Ys0=[Ys_l;Cls_su];
     acc_slu = length(find(Ys0==[Ys_l;Ys_u]))/length(Ys0)*100;
     if (t==5)
        accFSDA_s(1,iter)=acc_slu;
     end
     if (t==5)
        fprintf('Grap_source+acc_s=%0.1f\n',full(acc_slu));
     end
     %%  update target soft labels
     delta_t=0.01;
     Ct=At*Gt;
     kk=20;
     [Cls] = GraphClassifier4([Zs_l,Zs_u],Zt,Ys0,delta_t,At,Ct,kk);
     Cls_tu=Cls;
     acc_tu = length(find(Cls_tu==Yt))/length(Yt)*100;
     if (t==5)
        accFSDA_t(1,iter)=acc_tu;
     end
     if (t==5)
        fprintf('Grap_target+acc_t=%0.1f\n',full(acc_tu));
     end
     
     Yt0=Cls_tu;
     Fs=LabelFormat(Ys0);
     Ft=LabelFormat(Yt0);
     
end
if (iter-wrg)==10
    break;
end
end
accTCA_s=accTCA_s(IIDD);
accTCA_t=accTCA_t(IIDD);
accJDA_s=accJDA_s(IIDD);
accJDA_t=accJDA_t(IIDD);
accBDA_s=accBDA_s(IIDD);
accBDA_t=accBDA_t(IIDD);
accVDA_s=accVDA_s(IIDD);
accVDA_t=accVDA_t(IIDD);
accJGSA_s=accJGSA_s(IIDD);
accJGSA_t=accJGSA_t(IIDD);
accMEDA_s=accMEDA_s(IIDD);
accMEDA_t=accMEDA_t(IIDD);
accFSDA_s=accFSDA_s(IIDD);
accFSDA_t=accFSDA_t(IIDD);

res1=[mean(accTCA_s),mean(accJDA_s),mean(accBDA_s),mean(accVDA_s),mean(accJGSA_s),mean(accMEDA_s),mean(accFSDA_s)];
res1=roundn(res1,-1);
res2=[mean(accTCA_t),mean(accJDA_t),mean(accBDA_t),mean(accVDA_t),mean(accJGSA_t),mean(accMEDA_t),mean(accFSDA_t)];
res2=roundn(res2,-1);
res3=[std(accTCA_s,0,2),std(accJDA_s,0,2),std(accBDA_s,0,2),std(accVDA_s,0,2),std(accJGSA_s,0,2),std(accMEDA_s,0,2),std(accFSDA_s,0,2)];
res3=roundn(res3,-1);
res4=[std(accTCA_t,0,2),std(accJDA_t,0,2),std(accBDA_t,0,2),std(accVDA_t,0,2),std(accJGSA_t,0,2),std(accMEDA_t,0,2),std(accFSDA_t,0,2)];
res4=roundn(res4,-1);
res1234=[res1,res2;res3,res4];
RES=[RES;res1234];
end