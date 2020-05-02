function [acc_s,acc_t] = JGSA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options);

% Joint Geometrical and Statistical Alignment for Visual Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
% Jing Zhang, Wanqing Li, Philip Ogunbona.

alpha = options.alpha;
mu = options.mu;
beta = options.beta;
gamma = options.gamma;
ker = options.ker;
k = options.k;
T = options.T;
Xs = [Xs_l,Xs_u];
m = size(Xs,1);
ns = size(Xs,2);
nt = size(Xt,2);

class = unique(Ys_l);
C = length(class);

knn_model = fitcknn(Xs_l',Ys_l,'NumNeighbors',1);
Ys_pseudo = knn_model.predict(Xs_u');
Ys_pseudo = [Ys_l;Ys_pseudo];
knn_model = fitcknn([Xs_l';Xs_u'],Ys_pseudo,'NumNeighbors',1);
Yt_pseudo = knn_model.predict(Xt');

if strcmp(ker,'primal')
    
    %--------------------------------------------------------------------------

    for t = 1:T
        
        % compute LDA
        dim = size(Xs,1);
        meanTotal = mean(Xs,2);

        Sw = zeros(dim, dim);
        Sb = zeros(dim, dim);
        for i=1:C
            Xi = Xs(:,find(Ys_pseudo==class(i)));
            meanClass = mean(Xi,2);
            Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
            Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
            Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
        end
        P = zeros(2*m,2*m);
        P(1:m,1:m) = Sb;
        Q = Sw;
    
        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys_pseudo,Yt_pseudo,C);

        Ts = Xs*Ms*Xs';
        Tt = Xt*Mt*Xt';
        Tst = Xs*Mst*Xt';
        Tts = Xt*Mts*Xs';
        
        % Construct centering matrix
        Ht = eye(nt)-1/(nt)*ones(nt,nt);
        
        X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
        H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];

        Smax = mu*X*H*X'+beta*P;
        Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
                Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');
        A = W(1:m, :);
        Att = W(m+1:end, :);

        Zs = A'*Xs;
        Zt = Att'*Xt;
        
        if T>1
            ns_l = size(Xs_l,2);
            Zs_l = Zs(:,1:ns_l);
            Zs_u = Zs(:,ns_l+1:end);
            knn_model = fitcknn(Zs_l',Ys_l,'NumNeighbors',1);
            Ys_pseudo = knn_model.predict(Zs_u');
            Ys_pseudo = [Ys_l;Ys_pseudo];
            knn_model = fitcknn([Zs_l';Zs_u'],Ys_pseudo,'NumNeighbors',1);
            Yt_pseudo = knn_model.predict(Zt'); 
            acc_s = length(find(Ys_pseudo==[Ys_l;Ys_u]))/length(Ys_pseudo); 
            acc_t = length(find(Yt_pseudo==Yt))/length(Yt);
        end
    end
else
    
    Xst = [Xs, Xt];   
    nst = size(Xst,2); 
    [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);
   %--------------------------------------------------------------------------
    % compute LDA
    dim = size(Ks,2);
    C = length(class);
    meanTotal = mean(Ks,1);

    Sw = zeros(dim, dim);
    Sb = zeros(dim, dim);
    for i=1:C
        Xi = Ks(find(Ys==class(i)),:);
        meanClass = mean(Xi,1);
        Hi = eye(size(Xi,1))-1/(size(Xi,1))*ones(size(Xi,1),size(Xi,1));
        Sw = Sw + Xi'*Hi*Xi; % calculate within-class scatter
        Sb = Sb + size(Xi,1)*(meanClass-meanTotal)'*(meanClass-meanTotal); % calculate between-class scatter
    end
    P = zeros(2*nst,2*nst);
    P(1:nst,1:nst) = Sb;
    Q = Sw;        

    for t = 1:T

        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);
        
        Ts = Ks'*Ms*Ks;
        Tt = Kt'*Mt*Kt;
        Tst = Ks'*Mst*Kt;
        Tts = Kt'*Mts*Ks;

        K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
        Smax =  mu*K'*K+beta*P;
        
        Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
                Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), k, 'LM');
        W = real(W);

        A = W(1:nst, :);
        Att = W(nst+1:end, :);

        Zs = A'*Ks';
        Zt = Att'*Kt';

        if T>1
            Yt0 = knnclassify(Zt',Zs',Ys,1);  
            acc = length(find(Yt0==Yt))/length(Yt); 
            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
        end
    end
end
    
Xs = Zs;
Xt = Zt;

