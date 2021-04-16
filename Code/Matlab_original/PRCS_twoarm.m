function regret = PRCS_twoarm(mu,M,T,gridType)
    if strcmp(gridType,'minimax')
        a = T^(1/(2 - 2^(1-M))); TGrid = np.floor(a.^(2.-1./2.^(0:M-1)));...,
        TGrid(M) = T; TGrid = [0,TGrid]; % minimax batch grids
    elseif strcmp(gridType,'geometric')
        b = T^(1/M); TGrid = floor(b.^(1:M)); TGrid(M) = T; ...,
        TGrid = [0,TGrid]; % adaptive batch grids
    end
    
    pullnumber = round(TGrid(2)/2);
    regret = pullnumber * (mu(1) - mu(2)); 
    reward = sum([randn(pullnumber,1) + mu(1), randn(pullnumber,1) + mu(2) ], 1);
    opt = 0; 
    
    for m = 2 : M
        t = TGrid(m); 
        thres = sqrt(4 * log(2*T/t) / t); 
        if opt == 0          
            if (reward(1) - reward(2))/t > thres
                opt = 1;
            elseif (reward(2) - reward(1))/t > thres
                opt = 2;
            else
                cur_number = round((TGrid(m+1) - TGrid(m))/2); 
                pullnumber = pullnumber + cur_number; 
                reward = reward + sum([randn(cur_number,1) + mu(1), ...
                    randn(cur_number,1) + mu(2) ]);
                regret = regret + cur_number * (mu(1) - mu(2));
            end
        end
        
        if opt == 2
            regret = regret + (TGrid(m+1) - TGrid(m)) * (mu(1) - mu(2)); 
        end
        
        if m == (M-1)
            if reward(1) > reward(2)
                opt = 1; 
            else
                opt = 2;
            end
        end
    end
end