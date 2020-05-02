function [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma)

Xst = [Xs, Xt];   
ns = size(Xs,2);
nt = size(Xt,2);
nst = size(Xst,2); 
Kst0 = km_kernel(Xst',Xst',ker,gamma);
Ks0 = km_kernel(Xs',Xst',ker,gamma);
Kt0 = km_kernel(Xt',Xst',ker,gamma);

oneNst = ones(nst,nst)/nst;
oneN=ones(ns,nst)/nst;
oneMtrN=ones(nt,nst)/nst;
Ks=Ks0-oneN*Kst0-Ks0*oneNst+oneN*Kst0*oneNst;
Kt=Kt0-oneMtrN*Kst0-Kt0*oneNst+oneMtrN*Kst0*oneNst;
Kst=Kst0-oneNst*Kst0-Kst0*oneNst+oneNst*Kst0*oneNst;