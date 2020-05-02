function [acc_s,acc_t] = MEDA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
    %fprintf('MEDA starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end

    % Manifold feature learning
    Xs = [Xs_l;Xs_u];
    ns_l = size(Xs_l,1);
    ns_u = size(Xs_u,1);
    
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
    Xt = double(Xt_new');
    Xs_l = Xs(:,1:ns_l);
    Xs_u = Xs(:,ns_l+1:end);
    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Yt));
    
   %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));
    
    % Generate soft labels for the target domain
    knn_model = fitcknn(Xs_l',Ys_l,'NumNeighbors',1);
    Ys_pseudo = knn_model.predict(Xs_u');
    Ys_pseudo = [Ys_l;Ys_pseudo];
    knn_model = fitcknn([Xs_l';Xs_u'],Ys_pseudo,'NumNeighbors',1);
    Yt_pseudo = knn_model.predict(Xt');

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    E = diag(sparse([ones(n,1);zeros(m,1)]));

    for t = 1 : options.T
        YY = [];
        for c = 1 : C
            YY = [YY,Ys_pseudo==c];
        end
        YY = [YY;zeros(m,C)];
        % Estimate mu
        mu = estimate_mu(Xs',Ys_pseudo,Xt',Yt_pseudo);
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys_pseudo));
        N = 0;
        for c = reshape(unique(Ys_pseudo),1,length(unique(Ys_pseudo)))
            e = zeros(n + m,1);
            e(Ys_pseudo == c) = 1 / length(find(Ys_pseudo == c));
            e(n + find(Yt_pseudo == c)) = -1 / length(find(Yt_pseudo == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);

        %% Compute accuracy
        Ys_pseudo = Cls(1:n);
        Yt_pseudo = Cls(n+1:end);
        acc_s = numel(find(Cls(1:n)==[Ys_l;Ys_u])) / n;
        acc_t = numel(find(Cls(n+1:end)==Yt)) / m;
    end
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end