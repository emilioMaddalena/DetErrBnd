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

% ground-truth (Tinkerbell)
f = @(x1,x2) x1.^2 - x2.^2 + 0.8*x1 - 0.6*x2;

% domain
x1min = -5; x1max = 5;
x2min = -5; x2max = 5;

% Ground-truth RKHS norm estimate
% See. "Robust Uncertainty Bounds in Reproducing Kernel Hilbert Spaces: 
% A Convex Optimization Approach", Appendix A
Gamma = 196.1; 

%%
%%%%%%%%%%%%%%%%%%
% Collect samples
%%%%%%%%%%%%%%%%%%

% num of samples
N = 625;

% (uniform) noise bound
noise = [-0.5; 0.5]; 

% sampling strategy
meshSample = true;
rndSample = false;

if meshSample
    
    gran1 = (x1max-x1min)/(sqrt(N)-1);
    gran2 = (x2max-x2min)/(sqrt(N)-1);
    [X1,X2] = meshgrid([x1min:gran1:x1max],[x2min:gran2:x2max]);
    
elseif rndSample
    
    X1 = rand(N,1).*(x1max-x1min)+x1min;
    X2 = rand(N,1).*(x2max-x2min)+x2min;
    
end

% noise-less outputs
fX = f(X1,X2); 
fX = fX(:);

% noisy outputs
del = rand(N,1) * diff(noise) + noise(1);
y = fX + del;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel model and bounds
%%%%%%%%%%%%%%%%%%%%%%%%%%

% stack features 
Z = [X1(:) X2(:)];

% kernel matrix + jitter
K = kernel(Z,Z);
K = K + 0.00000001*eye(N); 

% KRR model
lam = 1e-5; 
krr = @(z) ((K+(N*lam)*eye(N))\y)'*kernel(Z,z);

% calculating norms
s_tilde_norm = sqrt(y'*(K\y));

% calculating Delta
delta = sdpvar(N,1);
constr = noise(1) <= delta <= noise(2);
cost = -(delta'/K)*delta + 2*(y'/K)*delta;
a = optimize(constr, -cost, sdpsettings('solver', 'gurobi', 'verbose', 0));
Delta = value(cost);

% Defining the several terms (see eq. 15)
% |s^*(x) - f(x)| <= P(x) * sqrt(Gamma^2 + Delta - ||s^tilde||^2) + p(x) + q(x)
%
% OBS. Since the noise bound is a scalar, the p(x) term simplifies to the
% Lebesgue function (see Sec. 4.2)

P = @(z) sqrt(real(diag(kernel(z,z) - (kernel(z,Z)/K)*kernel(Z,z))));
p = @(z) (noise(2) * vecnorm((kernel(z,Z)/K)', 1))';
q = @(z) (y' * ((K+(1/(N*lam))*K*K)\kernel(Z,z)))';

bound = @(z) P(z).*sqrt(Gamma^2 + Delta - s_tilde_norm^2) + abs(p(z)) + abs(q(z));

%%
%%%%%%%%%%%
% Plotting
%%%%%%%%%%%

figure
set(gcf,'color','w');
set(gcf,'Position',[100 100 1200 400])

% mesh
P = 1000; % increase to 10000 for higher resolution
gran1 = (x1max - x1min)/(sqrt(P) - 1);
gran2 = (x2max - x2min)/(sqrt(P) - 1);
[x1, x2] = meshgrid([x1min:gran1:x1max], [x2min:gran2:x2max]);
z = [x1(:) x2(:)];

% nominal model and error envelope
subplot(1, 2, 1); 
surf(x1, x2, reshape(krr(z), size(x1)) + reshape(real(bound(z)),size(x1)), 'FaceColor', 'y', 'FaceAlpha', 0.16, 'linestyle', '-', 'EdgeAlpha', 0.16); hold on
surf(x1, x2, reshape(krr(z), size(x1)) - reshape(real(bound(z)),size(x1)), 'FaceColor', 'y', 'FaceAlpha', 0.16, 'linestyle', '-', 'EdgeAlpha', 0.16);
axis([-5 5 -5 5 -40 40]); xticks([-5 0 5]); yticks([-5 0 5]); zticks([-40 -20 0 20 40]); view(-37.5,20.92)
set(gca, 'FontName', 'Latin Modern Roman')
xlabel('$x_1$', 'FontName', 'Helvetica', 'Interpreter', 'latex', 'FontSize', 15); 
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 15); 
zlabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 15)
surf(x1, x2, reshape(krr(z), size(x1)), 'FaceColor', '#0072BD', 'FaceAlpha', 0.8, 'linestyle', 'none');
ax = gca; ax.FontSize = 14;
title('Nominal model and upper/lower error bounds', 'Interpreter', 'latex')

% magnitude of error bounds
subplot(1, 2, 2); 
surf(x1, x2, reshape(real(bound(z)), size(x1)), 'linestyle', 'none');
axis('tight'); xticks([-5 0 5]); yticks([-5 0 5]); zticks(); view(90, 90)
xlabel('$x_1$', 'FontName', 'Helvetica', 'Interpreter', 'latex', 'FontSize', 15); 
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 15); 
zlabel('$e(x)$', 'Interpreter', 'latex', 'FontSize', 15)
ax = gca; ax.FontName = 'Latin Modern Roman'; ax.FontSize = 14;
title('Error bounds magnitude', 'Interpreter', 'latex')

% EOF