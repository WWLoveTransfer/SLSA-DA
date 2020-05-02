function F = GeneralSSL(A, T, is_normalize)

n = size(A,2);
ns = length(find(sum(T,2) == 1));
D = sum(A,2);
invD  = spdiags(1./D,0,n,n);
if is_normalize == 1 % normalized weights
    D1 = sqrt(invD);
    norA = D1*A*D1; 
    norD = sum(norA,2);
end

L=spdiags(norD,0,n,n)-norA;
Fs=T(1:ns,:);
Lst=L(1:ns,ns+1:end);
Lts=L(ns+1:end,1:ns);
Ltt=L(ns+1:end,ns+1:end);
F=-(2*Ltt)\(Lts*Fs+Lst'*Fs);