% Non-nagetive spectral clustering: min_{Q>=0} trace(Q'*L*Q) + r*trace((Q'*Q-I)'(Q'*Q-I))
function [Gt] = SolveG2(As, At, Gs, Gt, Ft, mu2)

ITER = 100;
if nargout > 1
    obj = zeros(ITER,1);
end;
G_pos1=At'*At;
G_pos1_pos=(abs(G_pos1)+G_pos1)/2;
G_pos1_neg=(abs(G_pos1)-G_pos1)/2;
G_pos2=mu2*At'*At;
G_pos2_pos=(abs(G_pos2)+G_pos2)/2;
G_pos2_neg=(abs(G_pos2)-G_pos2)/2;
G_neg1=At'*As;
G_neg1_pos=(abs(G_neg1)+G_neg1)/2;
G_neg1_neg=(abs(G_neg1)-G_neg1)/2;
G_neg2=mu2*At'*At;
G_neg2_pos=(abs(G_neg2)+G_neg2)/2;
G_neg2_neg=(abs(G_neg2)-G_neg2)/2;
 
for iter = 1:ITER
    obj(iter)=trace((As*Gs-At*Gt)*(As*Gs-At*Gt)')+mu2*trace((At-At*Gt*Ft)*(At-At*Gt*Ft)');
    if iter>1
        noi=obj(iter-1)-obj(iter);
        if noi<1e-4
        break;
        end
    end
    S=(G_pos1_neg*Gt+G_pos2_neg*Gt*Ft*Ft'+G_neg1_pos*Gs+G_neg2_pos*Ft'+eps)./...
      (G_pos1_pos*Gt+G_pos2_pos*Gt*Ft*Ft'+G_neg1_neg*Gs+G_neg2_neg*Ft'+eps);
    S=sqrt(S);
    Gt = Gt.*S;
    Gt = Gt*diag(sparse(1./sum(Gt,1)));
end
%Gt = Gt*diag(sparse(1./sum(Gt,1))); 