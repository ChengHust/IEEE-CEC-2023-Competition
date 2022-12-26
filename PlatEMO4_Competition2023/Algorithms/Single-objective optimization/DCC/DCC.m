classdef DCC < ALGORITHM
% <single> <real> <large/none>
% Dynamic cooperative coevolution for large scale optimization


%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            [G0,Gr,Gd,K] = deal(100,800,200,ceil(Problem.D/50));
            para = {[],zeros(Problem.N,1) + 0.5,zeros(Problem.N,1) + 0.5,1};
            Population  = Problem.Initialization();

            %% Obtain the lambda matrix based on DG2
            [lambda, ~,~] = ism(Problem);
            H = zeros(1,Problem.D);
            for i = 1 : Gr
                Groups = [RandomGrouping(Problem.D,K)];
                for k = 1 : K
                    flbest = Population.best.objs;
                    Population = Optimizer(Problem,Problem.N,non-PA);
                    H = [H;(Groups==k).*abs(flbest-Population.best.objs)];
                    [Population,para] = subSHADE(Problem,Population,Population.best,Groups==k,para);
                end
            end
            %% Optimize
            while Algorithm.NotTerminated(Population)
                for g = 1 : Gd
                    flbest = Population.best.objs;
                    xc = DynamicGrouping(Problem,H,50,lambda);
                    Population = Optimizer(Problem,Problem.N*G0,SSPA);
                    H = [H;(Groups==k).*abs(flbest-Population.best.objs)];
                end
            end
        end
    end
end

function xc = DynamicGrouping(Problem,H,D,lambda)
    N1 = min(100,Problem.D); 
    N2 = ceil(0.02*Problem.D);
    A  = mean(H,1);
    xc = [];
    [~,order] = sort(A,'descend');
    for i = 1 : N1
        [~,rank] = sort(lambda(i,:),'descend');
        for j = 1 : N2
            label = order(i);
            if ~any(xc==label)&&(length(xc)<=200)
                xc = [xc,label];
            end
            for k = 1 : D
                x_temp = rank(k);
                if ~any(xc==x_temp)&&(length(xc)<=200)
                    xc = [xc,x_temp];
                    break;
                end
            end
        end
    end
    xc = unique(xc);
end

function index = RandomGrouping(D,K)
    n = floor(D/K);
    index = [];
    for i = 1:K-1
       index = [index, ones(1,n).*i];
    end
    index = [index,ones(1,D-size(index,2)).*K];
    index = index(randperm(length(index)));
end

function [lambda, evaluations,POP] = ism(Problem)
    POP = []; D = Problem.D;
    center = 0.5 * (Problem.upper + Problem.lower);
    fhat_archive = NaN(D, 1);
    [f_archive,delta1,delta2,lambda] = deal(NaN(D));
    p1  = Problem.lower;
    fp1 = SOLUTION(p1).objs;
    for i=1:D-1
        if(~isnan(fhat_archive(i)))
            fp2 = fhat_archive(i);
        else
            X   = Replace(p1,center,i);
            POP = [POP,X];
            fp2 = X.objs;
            fhat_archive(i) = fp2;
        end
        for j=i+1:D
            if(~isnan(fhat_archive(j)))
                fp3 = fhat_archive(j);
            else
                X   = Replace(p1,center,j);
                POP = [POP,X];
                fp3 = X.objs;
                fhat_archive(j) = fp3;
            end
            X   = Replace(p1,center,[i,j]);
            POP = [POP,X];
            fp4 = X.objs;
            f_archive(i, j) = fp4;
            f_archive(j, i) = fp4;
            d1 = fp2 - fp1;
            d2 = fp4 - fp3;
            delta1(i, j) = d1;
            delta2(i, j) = d2;
            lambda(i, j) = abs(d1 - d2);
        end
    end
    evaluations.base = fp1;
    evaluations.fhat = fhat_archive;
    evaluations.F    = f_archive;
end


