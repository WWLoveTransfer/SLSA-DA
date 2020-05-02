% Non-nagetive spectral clustering: min_{Q>=0} trace(Q'*L*Q) + r*trace((Q'*Q-I)'(Q'*Q-I))
function [Gs] = SolveG1(As, At, Gs, Gt, Fs, mu1)

ITER = 100;
if nargout > 1
    obj = zeros(ITER,1);
end;

G_pos1=As'*As;
G_pos1_pos=(abs(G_pos1)+G_pos1)/2;
G_pos1_neg=(abs(G_pos1)-G_pos1)/2;
G_pos2=mu1*As'*As;
G_pos2_pos=(abs(G_pos2)+G_pos2)/2;
G_pos2_neg=(abs(G_pos2)-G_pos2)/2;
G_neg1=As'*At;
G_neg1_pos=(abs(G_neg1)+G_neg1)/2;
G_neg1_neg=(abs(G_neg1)-G_neg1)/2;
G_neg2=mu1*As'*As;
G_neg2_pos=(abs(G_neg2)+G_neg2)/2;
G_neg2_neg=(abs(G_neg2)-G_neg2)/2;
    
for iter = 1:ITER
    obj(iter)=trace((As*Gs-At*Gt)*(As*Gs-At*Gt)')+mu1*trace((As-As*Gs*Fs)*(As-As*Gs*Fs)');
    if iter>1
        noi=obj(iter-1)-obj(iter);
        if noi<1e-4
        break;
        end
    end
    S=(G_pos1_neg*Gs+G_pos2_neg*Gs*Fs*Fs'+G_neg1_pos*Gt+G_neg2_pos*Fs'+eps)./...
      (G_pos1_pos*Gs+G_pos2_pos*Gs*Fs*Fs'+G_neg1_neg*Gt+G_neg2_neg*Fs'+eps);
    S=sqrt(S);
    Gs = Gs.*S;
    Gs = Gs*diag(sparse(1./sum(Gs,1)));
end
%Gs = Gs*diag(sparse(1./sum(Gs,1)));