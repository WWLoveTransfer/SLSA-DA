function [Xl, Xu, Yl, Yu, nonL_indx] = select_labeled(X, Y, num)

c = length(unique(Y));
Xl=[];
Yl=[];
Xu=[];
Yu=[];
nonL_indx=[];
for i=1:c
    Ind_c=find(Y==i);
    num_c=length(Ind_c);
    rand_Ind=randperm(num_c,num);
    rand_Ind_non=setdiff(linspace(1,num_c,num_c),rand_Ind);
    nonL_indx=[nonL_indx,Ind_c(rand_Ind_non)'];
    Xc=X(:,Ind_c(rand_Ind));
    Yc=Y(Ind_c(rand_Ind));
    Xl=[Xl,Xc];
    Yl=[Yl;Yc];
    Xc_non=X(:,Ind_c(rand_Ind_non));
    Yc_non=Y(Ind_c(rand_Ind_non));
    Xu=[Xu,Xc_non];
    Yu=[Yu;Yc_non];
end

end

