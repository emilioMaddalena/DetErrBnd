% Supplementary material for the paper:
% 'Deterministic error bounds for kernel-based learning techniques under bounded noise'
% Authors: E. T. Maddalena, P. Scharnhorst and C. N. Jones
%
% Squared-exponential kernel with a single lengthscale, i.e., no ARD

function output = kernel(x1, x2)

    % lengthscale
    l = 0.707; 
    
    output = exp(-dist(x1, x2').^2 / (2*l^2));

end