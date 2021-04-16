function regret = UCB1(mu,K,T)
    pullnumber = ones(K,1);
    averageReward = zeros(K,1);
    
    for t = 1 : K
        averageReward(t) = mu(t) + randn; 
    end
    
    for t = (K+1) : T
        UCB = averageReward + sqrt(2*log(T) ./ pullnumber);
        [~, pos] = max(UCB);
        weight = 1/(pullnumber(pos) + 1);
        averageReward(pos) = (1-weight) * averageReward(pos) ...
            + weight * (mu(pos) + randn); 
        pullnumber(pos) = pullnumber(pos) + 1;
    end
    
    regret = (mu(1) - mu(2:end)) * pullnumber(2:end); 
end