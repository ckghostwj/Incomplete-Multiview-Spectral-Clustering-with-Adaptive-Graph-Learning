% % % The code is written by Jie Wen, if you have any problems, 
% % % please don't hesitate to contact me: jiewen_pr@126.com
 
% % % If you find the code is useful, please cite the following reference:
% % % Jie Wen, Yong Xu, Hong Liu, Incomplete Multi-view Spectral Clustering with
% % % Adaptive Graph Learning, IEEE Transactions on Cybenetics, 2018.
% % % 
% % % Note: tune parapeters lambda1, lambda2, lambda3, to achieve the best results
% % % 
% % % Influneced by the initilzation of k-means or the matlab versions, 
% % % the clustering results may have little deviations to the reuslts reported in the paper.
clear;
clc

Dataname = 'bbcsport4vbigRnSp';
Datafold = [Dataname,'_55percent_all_missing','.mat'];
lambda1  = 0.01;
lambda2  = 100000;
lambda3  = 0.1;


f = 3
load(Dataname);
load(Datafold);
ind_folds = folds{f};
truthF = truth;  
numClust = length(unique(truthF));
num_view = length(X);
for iv = 1:num_view
    X1 = X{iv}';
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];       
    Y{iv} = X1';            
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;                                               
end
clear X X1 W1
X = Y;
clear Y      
for iv = 1:num_view
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 3;
    options.WeightMode = 'Binary';
    Z1 = constructW(X{iv}',options);
    Z_ini{iv} = full(Z1);
    clear Z1;
end

for iv = 1:num_view
    invXX{iv} = inv(X{iv}'*X{iv}+2*eye(size(X{iv},2)));
end
F_ini = solveF(Z_ini,G,numClust);
U_ini = solveU(F_ini,numClust);

max_iter = 100;
miu = 0.01;
rho = 1.1;

[U] = IMSAGL(X,G,Z_ini,F_ini,invXX,numClust,lambda1,lambda2,lambda3,miu,rho,max_iter);
new_F = U;
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 
repeat = 5;
for iter_c = 1:repeat
    pre_labels    = kmeans(real(new_F),numClust,'emptyaction','singleton','replicates',20,'display','off');
    result_LatLRR = ClusteringMeasure(truthF, pre_labels);       
    AC(iter_c)    = result_LatLRR(1)*100;
    MIhat(iter_c) = result_LatLRR(2)*100;
    Purity(iter_c)= result_LatLRR(3)*100;
end
mean_ACC = mean(AC)
mean_NMI = mean(MIhat)
mean_PUR = mean(Purity)

