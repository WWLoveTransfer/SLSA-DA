function [acc_s,acc_t] = TCA(Xs_l,Xs_u,Xt,Ys_l,Ys_u,Yt,options)
                 
    lambda = options.lambda;
    dim = options.dim;
    X = [Xs_l',Xs_u',Xt'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns_l = size(Xs_l,1);
    ns_u = size(Xs_u,1);
    ns = ns_l+ns_u;
	nt = size(Xt,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];

    %% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);
	%%% M0
	M = e * e';  %multiply C for better normalization
    M = M / norm(M,'fro');
    knn_model = fitcknn(Xs_l,Ys_l,'NumNeighbors',1);
    Ys_pseudo = knn_model.predict(Xs_u);
    Ys_pseudo = [Ys_l;Ys_pseudo];
    knn_model = fitcknn([Xs_l;Xs_u],Ys_pseudo,'NumNeighbors',1);
    Yt_pseudo = knn_model.predict(Xt);

    %% Iteration
        
        %% Calculation
        [A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
        Z = A'*X;
            
        %normalization for better classification performance
		Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs_l = Z(:,1:ns_l)';
        Zs_u = Z(:,ns_l+1:ns)';
        Zt = Z(:,ns+1:end)';
        
        knn_model = fitcknn(Zs_l,Ys_l,'NumNeighbors',1);
        Ys_pseudo = knn_model.predict(Zs_u);
        Ys_pseudo = [Ys_l;Ys_pseudo];
        knn_model = fitcknn([Zs_l;Zs_u],Ys_pseudo,'NumNeighbors',1);
        Yt_pseudo = knn_model.predict(Zt);
 
        acc_s = length(find(Ys_pseudo==[Ys_l;Ys_u]))/length(Ys_pseudo); 
        acc_t = length(find(Yt_pseudo==Yt))/length(Yt_pseudo); 
end