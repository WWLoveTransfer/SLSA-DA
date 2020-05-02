function F = GeneralSSL3(A, T, is_normalize, delta_s, Cs, Asu)

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

ITER = 100;
obj = zeros(ITER,1);
F=zeros(n-ns,size(T,2))+1/10;
% °¥
G_pos1=2*Ltt;
G_pos1_pos=(abs(G_pos1)+G_pos1)/2;
G_pos1_neg=(abs(G_pos1)-G_pos1)/2;
G_pos2=Lts*Fs;
G_pos2_pos=(abs(G_pos2)+G_pos2)/2;
G_pos2_neg=(abs(G_pos2)-G_pos2)/2;
G_pos3=Lst'*Fs;
G_pos3_pos=(abs(G_pos3)+G_pos3)/2;
G_pos3_neg=(abs(G_pos3)-G_pos3)/2;
G_pos4=2*delta_s*Cs'*Cs;
G_pos4_pos=(abs(G_pos4)+G_pos4)/2;
G_pos4_neg=(abs(G_pos4)-G_pos4)/2;
G_neg1=2*delta_s*Asu'*Cs;
G_neg1_pos=(abs(G_neg1)+G_neg1)/2;
G_neg1_neg=(abs(G_neg1)-G_neg1)/2;

for iter = 1:ITER
    obj(iter)=trace(F'*Ltt*F)+trace(Fs'*Lst*F)+trace(F'*Lts*Fs)+delta_s*trace((Asu-Cs*F')*(Asu-Cs*F')');
    
    S=(G_pos1_neg*F+G_pos2_neg+G_pos3_neg+F*G_pos4_neg+G_neg1_pos+eps)./...
      (G_pos1_pos*F+G_pos2_pos+G_pos3_pos+F*G_pos4_pos+G_neg1_neg+eps);
    S=sqrt(S);
    F = F.*S;
end


