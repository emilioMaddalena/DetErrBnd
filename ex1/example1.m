% Supplementary material for the paper:
% 'Deterministic error bounds for kernel-based learning techniques under bounded noise'
% Authors: E. T. Maddalena, P. Scharnhorst and C. N. Jones
%
% Dependencies: Yalmip and Gurobi 
% If you don't have Gurobi, pick your favorite solver instead

%%
%%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%%

clc
clear all

% ground-truth
alpha = [-1; 3.5; 1.6; 6];
centers = [0; 2; 3; 5];
f = @(x) alpha(1)*kernel(x, centers(1)) + ...
         alpha(2)*kernel(x, centers(2)) + ...
         alpha(3)*kernel(x, centers(3)) + ...
         alpha(4)*kernel(x, centers(4));

% domain
xmin = -4; xmax = 10;

% exact RKHS norm can be computed (see eq. 5)
Gamma = sqrt(alpha'*kernel(centers,centers)*alpha); 

% but we will use the overestimate
Gamma = 9;

%%%%%%%%%%%%%%%%%%%%%
% Collecting samples
%%%%%%%%%%%%%%%%%%%%%

% num of samples
% N = 100 calls for jitter!
N = 20; 
X = linspace(xmin, xmax, N)';
    
% noise-less outputs
fX = f(X); 

% (uniform) noise bound
noise = [-0.15; 0.15]; 

% noisy outputs
del = rand(N,1) * diff(noise) + noise(1);
y = fX + del;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% KRR model and bounds
%%%%%%%%%%%%%%%%%%%%%%%%%%

% kernel matrix
K = kernel(X, X);

% Kernel matrix jitter
% All expression are sensitive to the ill-conditioning of the
% kernel matrix, especially the 'alternative bounds'
% K = K + 0.005*eye(N); 

% KRR model
lam = 0.001; 
krr = @(x) (((K+(N*lam)*eye(N))\y)'*kernel(X,x))';

% Calculating Delta
delta = sdpvar(N,1);
constr = noise(1) <= delta <= noise(2);
cost = -(delta'/K)*delta + 2*(y'/K)*delta;
optimize(constr, -cost, sdpsettings('solver', 'gurobi', 'verbose', 0));
Delta = value(cost);

% Defining the several terms (see eq. 15)
% |s^*(x) - f(x)| <= P(x) * sqrt(Gamma^2 + Delta - ||s^tilde||^2) + p(x) + q(x)
%
% OBS. Since the noise bound is a scalar, the p(x) term simplifies to the
% Lebesgue function (see Sec. 4.2)

s_tilde_norm = sqrt(y'*(K\y));

P = @(x) sqrt(diag(kernel(x,x) - (kernel(x,X)/K)*kernel(X, x)));%sqrt(real(diag(kernel(x,x) - (kernel(x,X)/K)*kernel(X, x))));
p = @(x) (noise(2) * vecnorm((kernel(x,X)/K)', 1))';
q = @(x) (y' * ((K+(1/(N*lam))*K*K)\kernel(X,x)))';

krr_bound = @(x) P(x).*sqrt(Gamma^2 + Delta - s_tilde_norm^2) + abs(p(x)) + abs(q(x));

%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVR model and bounds
%%%%%%%%%%%%%%%%%%%%%%%%%%

% nominal model
alpha = sdpvar(N,1);
constr = noise(1) <= K*alpha - y <= noise(2);
cost = alpha'*K*alpha;
optimize(constr, cost, sdpsettings('solver', 'gurobi', 'verbose', 0));
alpha = value(alpha);
svr_norm = sqrt(alpha'*K*alpha);
svr = @(x) kernel(X,x)'*alpha;

% bounds norms (see eq. 21)
r = @(x) abs((K*alpha - y)' * (K\kernel(X,x)))';
svr_bound = @(x) P(x).*sqrt(Gamma^2 - svr_norm^2) + abs(p(x)) + abs(r(x));

%%%%%%%%%%%%%%%%%%%%
% Alternative model 
%%%%%%%%%%%%%%%%%%%%

% nominal model
delta_tilde = max(abs(noise));
alt = @(x) kernel(x,X)*((K+(delta_tilde^2)*eye(N))\y);

% bounds norms (see eq. 27)
sigma = @(x) sqrt(real(diag(kernel(x,x) - (kernel(x,X)/(K + delta_tilde^2))*kernel(X, x))));
alt_bound = @(x) sigma(x).*sqrt(Gamma^2 - y'*((K+delta_tilde^2)\y) + N);

%%
%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%

figure; hold on
set(gcf,'color','w');
set(gcf,'Position', [100 100 1000 400])

blue = [0 0.4470 0.7410];
green = [0.4660, 0.6740, 0.1880];
yellow = [0.9290, 0.6940, 0.1250];

xx = linspace(xmin, xmax, 200)';

% envelopes
alt_upper_env = real(alt(xx) + alt_bound(xx));
alt_lower_env = real(alt(xx) - alt_bound(xx));
fill([xx; flipud(xx)], [alt_lower_env; flipud(alt_upper_env)], yellow, 'facealpha', 0.15, 'linewidth', 1.5)

svr_upper_env = svr(xx) + svr_bound(xx);
svr_lower_env = svr(xx) - svr_bound(xx);
fill([xx; flipud(xx)], [svr_lower_env; flipud(svr_upper_env)], green, 'facealpha', 0.4, 'linewidth', 1.5)

krv_upper_env = krr(xx) + krr_bound(xx);
krv_lower_env = krr(xx) - krr_bound(xx);
fill([xx; flipud(xx)], [krv_lower_env; flipud(krv_upper_env)], blue, 'facealpha', 0.9, 'linewidth', 1.5)

plot(xx, f(xx), 'k--', 'linewidth', 2); 
axis([-4 10 -2 7]); grid on

% nominal models
plot(xx, krr(xx), 'color', 'b', 'linewidth', 2);
plot(xx, svr(xx), 'color', green, 'linewidth', 2);
plot(xx, alt(xx), 'color', 'y', 'linewidth', 2);

% samples
scatter(X, y, 35, 'MarkerEdgeColor', 'w', 'MarkerFaceColor', 'k', 'LineWidth', 2);

set(gca, 'FontName', 'Latin Modern Roman')
xlabel('$x$', 'FontName', 'Helvetica', 'Interpreter', 'latex', 'FontSize', 15); 
ylabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 15); ax = gca; ax.FontSize = 14;
title('Samples, nominal models and deterministic error-bounds', 'Interpreter', 'latex')

% EOF