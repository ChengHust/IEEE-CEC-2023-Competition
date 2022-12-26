function [Population,xbest,ccm,para] = subSaNSDE2(Problem,Population,xbest,selected,para)
    [ccm,linkp,fp] = deal(para(1:3));
    [l1,l2,nl1,nl2,ns1,ns2,nf1,nf2] = deal(para(4:end));
    [subN,D] = size(Population.decs);
    OffDec   = zeros(subN, D);                                              % intermediate population of perturbed vectors
    rot      = 0: subN - 1;                                                 % rotating index array (size subN)
    %% Optimization
    for i = 1 : 1
        POPDEC = Population.decs;
        ci = randperm(4);
        a1 = randperm(subN);                     pm1 = POPDEC(a1, :);
        a2 = a1(rem(rot + ci(1), subN) + 1);     pm2 = POPDEC(a2, :);  
        a3 = a2(rem(rot + ci(2), subN) + 1);     pm3 = POPDEC(a3, :); 
        a4 = a3(rem(rot + ci(3), subN) + 1);     pm4 = POPDEC(a4, :);
        bm = repmat(xbest.decs,subN,1);
        if rem(i, 5) == 1
            cc = UpdateCC(subN,ccm);
        end
        fst1 = (rand(subN, 1) <= fp); fst2 = ~fst1;
        F  = repmat(GetF(subN,fp),1,D);
        mu = rand(subN,D) < repmat(cc,1,D);
        index  = all(mu==0,2);
        mu(index, ceil(D*rand(1,sum(index)))) = true;
        mua    = rand(subN, 1) <= linkp; mub = ~mua;
        cross  = bm(mub,:)-POPDEC(mub,:) + pm1(mub,:) - pm2(mub,:) + pm3(mub,:) - pm4(mub,:);
        OffDec(mub,:) = POPDEC(mub,:).*(1-mu(mub,:)) + (POPDEC(mub,:) + F(mub,:).*cross).*mu(mub,:);
        OffDec(mua,:) = POPDEC(mua,:).*(1-mu(mua,:)) + (pm3(mua,:) + F(mua,:).*(pm1(mua,:) - pm2(mua,:))).*mu(mua,:);

        %% Update the population
        OffSpring = SelectedEval(Problem,OffDec,xbest,selected);
        better    = OffSpring.objs <= Population.objs; worse = ~better;
        Population(better) = OffSpring(better);   
        l1  = l1 + sum(mua(better));     l2  = l2 + sum(mub(better));
        ns1 = ns1 + sum(fst1(better));   ns2 = ns2 + sum(fst2(better));
        nl1 = nl1 + sum(mua(worse));     nl2 = nl2 + sum(mub(worse));
        nf1 = nf1 + sum(fst1(worse));    nf2 = nf2 + sum(fst2(worse));
        [best, ibest] = min(Population.objs);
        if best < xbest.objs
            xbest = Population(ibest);
        end
        if rem(i, 24) == 1 && i ~= 1
            linkp = (l1 / (l1 + nl1)) / (l1 / (l1 + nl1) + l2 / (l2 + nl2));
            [l1,l2,nl1,nl2] = deal(1);
            fp = (ns1 * (ns2 + nf2)) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2));
            [ns1,ns2,nf1,nf2] = deal(1);
        end
    end
    para = [ccm,linkp,fp,l1,l2,nl1,nl2,ns1,ns2,nf1,nf2];
end

function F = GetF(subN,fp)
    R = rand(subN, 1) <= fp; 
    F  = abs(normrnd(0.5, 0.3, subN, 1).*R + normrnd(0, 1, subN, 1)./normrnd(0,1,subN, 1).*(1-R));
end

function cc = UpdateCC(subN,ccm)
        index = false;
        while sum(index) < subN
            cc = normrnd(ccm, 0.1, subN * 3, 1);
            index = (cc < 1) & (cc > 0);
        end
        cc = cc(index);
        cc = cc(1: subN);
end