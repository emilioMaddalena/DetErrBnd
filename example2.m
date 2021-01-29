% Supplementary material for the paper: "Deterministic Error Bounds for 
% Nonlinear Kernel Models Learned With Bounded Noise"
%
% Example 2
%% Preliminaries

clc
clear all
rng(10); % 10 2

% ground-truth
f = @(x1,x2) x1.^2 - x2.^2 + 0.8*x1 - 0.6*x2;

% domain
x1min = -5; x1max = 5;
x2min = -5; x2max = 5;

% defining the kernel
l = 1.62; 
kernel = @(x,y) exp(-dist(x,y').^2 / (2*l^2));

% estimating the ground-truth RKHS norm
Gamma = 196.1; % applying a safety factor

% Noise bound
noise = [-0.5; 0.5]; 

j = 1;

% sampling (mesh)
N = 625;
gran1 = (x1max-x1min)/(sqrt(N)-1);
gran2 = (x2max-x2min)/(sqrt(N)-1);
[X1,X2] = meshgrid([x1min:gran1:x1max],[x2min:gran2:x2max]);

% sampling (rand)
% N = 625;
% X1 = rand(N,1).*(x1max-x1min)+x1min;
% X2 = rand(N,1).*(x2max-x2min)+x2min;

% XX1 = [-5:1:5 -5:1:5 -5*ones(size(-5:1:5)) 5*ones(size(-5:1:5))]';
% XX2 = [-5*ones(size(-5:1:5)) 5*ones(size(-5:1:5)) -5:1:5 -5:1:5]';
% N = N + size(XX1,1);
% X1 = [X1; XX1(:)];
% X2 = [X2; XX2(:)];

% packing
Z = [X1(:) X2(:)];

% KRR regularization weight
lam = 1e-5; % reg weight

K = kernel(Z,Z);
K = K + 0.00000001*eye(N); % jitter

% exact measurements
fX = f(X1,X2); 
fX = fX(:);
% noise
rng(3)
del = rand(N,1) * diff(noise) + noise(1);
% noisy measurements
nfX = fX + del;

% calculating norms
s_noisy = sqrt(nfX'*(K\nfX));
Kkrr = K + (N*lam)*eye(N);
krr_norm = sqrt(nfX'*(Kkrr\K/Kkrr)*nfX);

s_tilde = @(z) (K\nfX)'*kernel(Z, z);
krr = @(z) ((K+(N*lam)*eye(N))\nfX )'*kernel(Z,z);

% SVR model
alpha = sdpvar(N,1);
constr = noise(1) <= K*alpha - nfX <= noise(2);
cost = alpha'*K*alpha;
a = optimize(constr,cost,sdpsettings('solver','mosek','verbose',0));
alpha = value(alpha);
svr_norm = sqrt(alpha'*K*alpha);
svr = @(z) kernel(Z,z)'*alpha;

% plotting mesh
P = 10000; %400; %10000;
gran1 = (x1max-x1min)/(sqrt(P)-1);
gran2 = (x2max-x2min)/(sqrt(P)-1);
[x1,x2] = meshgrid([x1min:gran1:x1max],[x2min:gran2:x2max]);

%gran1 = 4/(sqrt(P)-1);
%gran2 = 4/(sqrt(P)-1);
%[x1,x2] = meshgrid([-2.2:gran1:2.2],[-2.2:gran2:2.2]);

z = [x1(:) x2(:)];

%%%%%%%%%%%%%%%%%%
% Nominal models
%%%%%%%%%%%%%%%%%%

if false
% Ground-truth, interpolant, krr model
figure(1)
win = gcf;
subplot(1,4,1); hold 
h1 = surf(x1,x2,f(x1,x2)); plot3(X1(end-size(XX1,1)+1:end),X2(end-size(XX1,1)+1:end),nfX(end-size(XX1,1)+1:end), 'ro', 'markersize', 10);
title('Ground-truth'); axis('tight')
subplot(1,4,2); hold 
h2 = surf(x1,x2,reshape(s_tilde(z),size(x1))); %plot3(X1(:),X2(:),nfX, 'ro', 'markersize', 10);
title('Interpolant'); axis('tight')
subplot(1,4,3); hold on
h3 = surf(x1,x2,reshape(krr(z),size(x1))); 
%plot3(X1(:),X2(:),nfX, 'ro', 'markersize', 10);
title('KRR'); axis('tight')
subplot(1,4,4); hold on
h3 = surf(x1,x2,reshape(svr(z),size(x1))); 
%plot3(X1(:),X2(:),nfX, 'ro', 'markersize', 10);
title('SVR'); axis('tight')
end 
% bounds

% Calculating Delta
delta = sdpvar(N,1);
constr = noise(1) <= delta <= noise(2);
cost = (delta'/K)*delta - 2*(nfX'/K)*delta;
a = optimize(constr,cost,sdpsettings('solver','mosek','verbose',0));
delta = value(delta);
Delta = -(delta'/K)*delta + 2*(nfX'/K)*delta;

% Defining the several terms
pow = @(z) sqrt(real(diag(kernel(z,z)-(kernel(z,Z)/K)*kernel(Z,z))));
p = @(z) (noise(2) * vecnorm((kernel(z,Z)/K)',1) )';
q = @(z) (nfX' * ((K+(1/(N*lam))*K*K)\kernel(Z,z)))';

if (Gamma^2 + Delta - s_noisy^2) >= 0 
    bound = @(z) pow(z).*sqrt(Gamma^2 + Delta - s_noisy^2) + abs(p(z)) + abs(q(z));
    disp('pass')
else
    bound = @(z) sqrt(pow(z)).*sqrt(Gamma^2) + abs(p(z)) + abs(q(z));   
    disp('problems!')
end

% if (s_noisy^2 - Delta) >= 0 
%     disp('okay as an estimate!')
%     s_noisy^2
%     Delta
% else
%     disp('DANG :(')
% end

svr_bound = @(z) pow(z).*sqrt(Gamma^2 - svr_norm^2) + 2*p(z);

figure(2)
subplot(1,3,j); 
surf(x1,x2,reshape(real(bound(z)),size(x1)), 'linestyle', 'none');
xlabel('$x_1$', 'FontName', 'Helvetica', 'Interpreter', 'latex', 'FontSize', 15); ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 15); zlabel('$e(x)$', 'Interpreter', 'latex', 'FontSize', 15)
axis('tight'); xticks([-5 0 5]); yticks([-5 0 5]); zticks();
view(90, 90)
ax = gca;
ax.FontName = 'Latin Modern Roman'; ax.FontSize = 14;
set(gcf,'color','w');

disp('Average bound:')
i = (sum(real(bound(z))))/numel(x1)

%colormapeditor

%%
figure(3)
subplot(1,3,j); 
surf(x1,x2,reshape(krr(z),size(x1))+reshape(real(bound(z)),size(x1)),'FaceColor','y','FaceAlpha',0.16,'linestyle','-','EdgeAlpha',0.16); hold on
surf(x1,x2,reshape(krr(z),size(x1))-reshape(real(bound(z)),size(x1)),'FaceColor','y','FaceAlpha',0.16,'linestyle','-','EdgeAlpha',0.16);
axis([-5 5 -5 5 -40 40]); 
view(-37.5,20.92)
set(gca, 'FontName', 'Latin Modern Roman')
xticks([-5 0 5]); yticks([-5 0 5]); zticks([-40 -20 0 20 40]);
xlabel('$x_1$', 'FontName', 'Helvetica', 'Interpreter', 'latex', 'FontSize', 15); ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 15); zlabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 15)
%colormap summer; shading interp
surf(x1,x2,reshape(krr(z),size(x1)),'FaceColor','#0072BD','FaceAlpha',0.8,'linestyle','none');
ax = gca;
ax.FontSize = 14;
set(gcf,'color','w');
%%
set(gcf,'Position',[100 100 1800 400])
exportgraphics(gcf, 'test.pdf', 'ContentType', 'vector')

set(gcf,'Position',[100 100 1800 350])
export_fig 2a -png -q100 -m3

%% Comparing KRR and SVR

figure(3)
subplot(2,2,1); 
surf(x1,x2,reshape(real(bound(z)),size(x1)));
axis('tight'); 
colormap summer
shading interp

subplot(2,2,2); 
surf(x1,x2,reshape(real(svr_bound(z)),size(x1)));
axis('tight'); 
colormap summer
shading interp

subplot(2,2,3); 
surf(x1,x2,reshape(krr(z),size(x1))+reshape(real(bound(z)),size(x1)),...
    'linestyle','none','FaceColor','g','FaceAlpha',0.5); hold on
surf(x1,x2,reshape(krr(z),size(x1))-reshape(real(bound(z)),size(x1)),...
    'linestyle','none','FaceColor','g','FaceAlpha',0.5);
axis('tight'); view(-37.5,20.92); colormap summer; shading interp
surf(x1,x2,reshape(krr(z),size(x1)),'FaceColor','#0072BD','FaceAlpha',0.8,'linestyle','none');

%xticks([-4 -2 0 2 4]); yticks([-4 -2 0 2 4]); zticks([-2 0 2 4 6]);

subplot(2,2,4); 
surf(x1,x2,reshape(svr(z)+real(svr_bound(z)),size(x1)),...
    'linestyle','none','FaceColor','g','FaceAlpha',0.5); hold on
surf(x1,x2,reshape(svr(z)-real(svr_bound(z)),size(x1)),...
    'linestyle','none','FaceColor','g','FaceAlpha',0.5);
axis('tight'); view(-37.5,20.92); colormap summer; shading interp
surf(x1,x2,reshape(svr(z),size(x1)),'FaceColor','#0072BD','FaceAlpha',0.8,'linestyle','none');

%xticks([-4 -2 0 2 4]); yticks([-4 -2 0 2 4]); zticks([-2 0 2 4 6]);