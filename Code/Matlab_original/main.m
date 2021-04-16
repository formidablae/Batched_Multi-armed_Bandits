clear;clc;close all;

%% Parameters 
K = 3; T = 5e4; M = 3; 
K_set = round(logspace(log10(2),log10(20),6));
T_set = round(logspace(log10(500),log10(5e4),6)); 
M_set = 2:7; 
mu_max = 0.6; mu_min = 0.5; gamma = 0.5; m = 200; 

%% Experiments

% dependence on M
regretMinimax_M = zeros(m,length(M_set));
regretGeometric_M = zeros(m,length(M_set)); 
regretArithmetic_M = zeros(m,length(M_set)); 
regretUCB_M = zeros(m,1); 
mu = [mu_max, mu_min * ones(1,K-1)]; 
for iter = 1 : m
    regretUCB_M(iter) = UCB1(mu,K,T); 
    for iter_M = 1 : length(M_set)
        temp_M = M_set(iter_M); 
        regretMinimax_M(iter,iter_M) = BASEFunc(mu,K,T,temp_M,'minimax',gamma);
        regretGeometric_M(iter,iter_M) = BASEFunc(mu,K,T,temp_M,'geometric',gamma);
        regretArithmetic_M(iter,iter_M) = BASEFunc(mu,K,T,temp_M,'arithmetic',gamma);
    end
end
regretMinimax_M_mean = mean(regretMinimax_M) / T; 
regretGeometric_M_mean = mean(regretGeometric_M) / T; 
regretArithmetic_M_mean = mean(regretArithmetic_M) / T; 
regretUCB_M_mean = mean(regretUCB_M) / T; 

% dependence on K
regretMinimax_K = zeros(m,length(K_set));
regretGeometric_K = zeros(m,length(K_set)); 
regretArithmetic_K = zeros(m,length(K_set)); 
regretUCB_K = zeros(m,length(K_set)); 
for iter = 1 : m
    for iter_K = 1 : length(K_set)
        temp_K = K_set(iter_K); 
        mu = [mu_max, mu_min * ones(1, temp_K-1)]; 
        regretUCB_K(iter,iter_K) = UCB1(mu,temp_K,T); 
        regretMinimax_K(iter,iter_K) = BASEFunc(mu,temp_K,T,M,'minimax',gamma);
        regretGeometric_K(iter,iter_K) = BASEFunc(mu,temp_K,T,M,'geometric',gamma);
        regretArithmetic_K(iter,iter_K) = BASEFunc(mu,temp_K,T,M,'arithmetic',gamma); 
    end
end
regretMinimax_K_mean = mean(regretMinimax_K) / T; 
regretGeometric_K_mean = mean(regretGeometric_K) / T; 
regretArithmetic_K_mean = mean(regretArithmetic_K) / T; 
regretUCB_K_mean = mean(regretUCB_K) / T; 

% dependence on T 
regretMinimax_T = zeros(m,length(T_set));
regretGeometric_T = zeros(m,length(T_set)); 
regretArithmetic_T = zeros(m,length(T_set)); 
regretUCB_T = zeros(m,length(T_set));
mu = [mu_max, mu_min * ones(1, K-1)]; 
for iter = 1 : m
    for iter_T = 1 : length(T_set)
        temp_T = T_set(iter_T); 
        regretUCB_T(iter,iter_T) = UCB1(mu,K,temp_T)/temp_T; 
        regretMinimax_T(iter,iter_T) = BASEFunc(mu,K,temp_T,M,'minimax',gamma)/temp_T;
        regretGeometric_T(iter,iter_T) = BASEFunc(mu,K,temp_T,M,'geometric',gamma)/temp_T;
        regretArithmetic_T(iter,iter_T) = BASEFunc(mu,K,temp_T,M,'arithmetic',gamma)/temp_T; 
    end
end
regretMinimax_T_mean = mean(regretMinimax_T); 
regretGeometric_T_mean = mean(regretGeometric_T); 
regretArithmetic_T_mean = mean(regretArithmetic_T); 
regretUCB_T_mean = mean(regretUCB_T); 

% comparison with [PRCS16]
regretMinimax = zeros(m,length(M_set));
regretGeometric = zeros(m,length(M_set)); 
regretPRCSminimax = zeros(m,length(M_set));
regretPRCSgeometric = zeros(m,length(M_set)); 
regretUCB = zeros(m,1);
mu = [mu_max, mu_min]; 
for iter = 1 : m
    regretUCB(iter) = UCB1(mu,2,T);
    for iter_M = 1 : length(M_set)
        temp_M = M_set(iter_M); 
        regretMinimax(iter,iter_M) = BASEFunc(mu,2,T,temp_M,'minimax',gamma);
        regretGeometric(iter,iter_M) = BASEFunc(mu,2,T,temp_M,'geometric',gamma);
        
        regretPRCSminimax(iter,iter_M) = PRCS_twoarm(mu,temp_M,T,'minimax');
        regretPRCSgeometric(iter,iter_M) = PRCS_twoarm(mu,temp_M,T,'geometric');
    end
end
regretMinimax_mean = mean(regretMinimax) / T;
regretGeometric_mean = mean(regretGeometric) / T; 
regretPRCSminimax_mean = mean(regretPRCSminimax) / T;
regretPRCSGeometric_mean = mean(regretPRCSgeometric) / T; 
regretUCB_mean = mean(regretUCB) / T; 

% Figures
figure;
plot(M_set, regretMinimax_M_mean, 'bs-', 'MarkerFaceColor','b','linewidth', 2); hold on;
plot(M_set, regretGeometric_M_mean, 'ro--', 'MarkerFaceColor','r','linewidth', 2);
plot(M_set, regretArithmetic_M_mean, 'cv-.', 'MarkerFaceColor','c','linewidth',2);
plot(M_set, regretUCB_M_mean * ones(size(M_set)), 'k:', 'linewidth', 2); 
xticks(2:7); xlabel('$M$'); ylabel('Average regret');
legend('Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1');

figure; 
plot(K_set, regretMinimax_K_mean, 'bs-', 'MarkerFaceColor','b','linewidth', 2); hold on; 
plot(K_set, regretGeometric_K_mean, 'ro--', 'MarkerFaceColor','r','linewidth', 2);
plot(K_set, regretArithmetic_K_mean, 'cv-.', 'MarkerFaceColor','c','linewidth',2);
plot(K_set, regretUCB_K_mean, 'k:', 'linewidth', 2); 
xlabel('$K$'); ylabel('Average regret'); 
legend('Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1');

figure; 
semilogx(T_set, regretMinimax_T_mean, 'bs-', 'MarkerFaceColor','b','linewidth', 2); hold on; 
semilogx(T_set, regretGeometric_T_mean, 'ro--', 'MarkerFaceColor','r','linewidth', 2);
semilogx(T_set, regretArithmetic_T_mean, 'cv-.', 'MarkerFaceColor','c','linewidth',2);
semilogx(T_set, regretUCB_T_mean, 'k:', 'linewidth', 2); 
xlim([5e2,5e4]); xlabel('$T$'); ylabel('Average regret'); 
legend('Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1');

figure;
plot(M_set, regretMinimax_mean, 'bs-', 'MarkerFaceColor','b','linewidth', 2); hold on; 
plot(M_set, regretGeometric_mean, 'bs--', 'MarkerFaceColor','b','linewidth', 2);
plot(M_set, regretPRCSminimax_mean, 'ro-', 'MarkerFaceColor','r','linewidth', 2); 
plot(M_set, regretPRCSGeometric_mean, 'ro--', 'MarkerFaceColor','r','linewidth', 2);
plot(M_set, regretUCB_mean * ones(size(M_set)), 'k:', 'linewidth', 2); 
xticks(2:7); xlabel('$M$'); ylabel('Average regret'); 
legend('BaSE: Minimax Grid', 'BaSE: Geometric Grid', 'ETC: Minimax Grid', 'ETC: Geometric Grid', 'UCB1');
