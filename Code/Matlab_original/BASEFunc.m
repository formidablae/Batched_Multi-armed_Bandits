% function of BASE
% parameters
%     K: number of batches
%     TSeq: horizon
%     M: number of batches
%     b = T^(1/M); TGridAdaptive = floor(b.^(1:M));...,
%         TGridAdaptive = floor(TGridAdaptive/K) * K; TGridAdaptive(M) = T; ...,
%         TGridAdaptive = [0,TGridAdaptive]; % adaptive batch grids
%     a = T^(1/(2 - 2^(1-M))); TGridMinimax = floor(a.^(2.-1./2.^(0:M-1)));...,
%     TGridMinimax(M) = T; ...,
%     TGridMinimax = [0,TGridMinimax]; % minimax batch grids    
%     mu: batch mean
%     gamma: tunning parameter

function [regret, activeSet] = BASEFunc(mu, K, T, M, gridType, gamma) 
    % record
    regret = 0;
    if strcmp(gridType,'minimax')
        a = T^(1/(2 - 2^(1-M))); TGrid = floor(a.^(2.-1./2.^(0:M-1)));...,
        TGrid(M) = T; TGrid = [0,TGrid]; % minimax batch grids
    elseif strcmp(gridType,'geometric')
        b = T^(1/M); TGrid = floor(b.^(1:M)); TGrid(M) = T; ...,
        TGrid = [0,TGrid]; % geometric batch grids
    elseif strcmp(gridType,'arithmetic')
        TGrid = floor(linspace(0, T, M+1)); 
    end
  
    % initialization
    activeSet = ones(1,K); numberPull = zeros(1,K); averageReward = zeros(1,K);
    
    for i = 2:M+1
        availableK = sum(activeSet);
        pullNumber = max(floor((TGrid(i) - TGrid(i-1))/availableK), 1);
        TGrid(i) = availableK * pullNumber + TGrid(i-1);
        for j = find(activeSet == 1)
            averageReward(j) = averageReward(j) * (numberPull(j)/(numberPull(j) ...,
                + pullNumber)) + (mean(randn(1,pullNumber)) + mu(j)) * ...,
                (pullNumber/(numberPull(j) + pullNumber));
            regret = regret + (pullNumber * (mu(1) - mu(j)));
            numberPull(j) = numberPull(j) + pullNumber; 
        end
        maxArm = max(averageReward(find(activeSet == 1)));
        for j = find(activeSet == 1)
            if ((maxArm - averageReward(j)) >= sqrt(gamma * log(T*K) / numberPull(j)))
                activeSet(j) = 0;
            end
        end
    end
end


