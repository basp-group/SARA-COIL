function [x,u,v] = PD_ghost_SARA(y, Phi, param)

% -------------------------------------------------------------------------
% COIL-SARA
% Primal dual algorithm for compressive optical imaging with a photonic
% lantern
% -------------------------------------------------------------------------
% Problem of interest: y = Phi * xtrue
% with - xtrue : original unknown image
%      - Phi   : measurement matrix
%                concatenation of the projection patterns
%      - y     : observations
% -------------------------------------------------------------------------
% Minimisation problem:
% Minimise || W Psit(x) ||_1 s.t.  x positive
%                             and  || y - Phi*x || <= epsilon
% -------------------------------------------------------------------------
% param contains:
%    - Nx, Ny    : dimension of the image of interest
%    - x0        : initialisation for the algorithm
%    - u,v       : initialisation for the dual variable - if available
%    - Psit, Psi : forward and backward operator for the sparistiy basis
%    - normPsi   : spectral norm of Psi
%    - normPhi   : spectral norm of Phi
%    - weights   : weights - if available
%    - lambda    : free parameter acting on the convergence speed
%    - stopcrit  : stopping criterion checking the relative change between
%                  two consecutive values of the objective function
%    - stopnorm  : stopping criterion checking the relative change between
%                  two consecutive iterates
%    - stopbound : tolerance for the l2 constraint
%    - NbIt      : number of iterations
%    - display   : number of iterations to display information
% -------------------------------------------------------------------------
% *************************************************************************
% *************************************************************************
% version 0.0
% November 2018
%
% Author: Audrey Repetti
% Contact: arepetti@hw.ac.uk
%
% For details on the method, refer to the article
% Compressive optical imaging with a photonic lantern
%
% For further details on primal-dual algorithm, refer to 
% L. Condat, ?A primal-dual splitting method for convex optimization  
% involving Lipschitzian, proximable and linear composite terms,? J.  
% Optimization Theory and Applications, vol. 158, no. 2, pp. 460-479, 2013
% *************************************************************************
% *************************************************************************


%% Initialisation

if ~isfield(param,'weights')
    weights = 1 ;
else
    weights = param.weights ;
end

Psit =@(x) param.Psit(reshape(x,param.Ny, param.Nx)) ;
Psi =@(x) reshape(param.Psi(x), param.Nx*param.Ny,1) ;

sigma = 0.99 / sqrt(param.normPsi + param.normPhi) ;
tau = 0.99 / ( sigma * (param.normPsi + param.normPhi) ) ;

if sigma*tau*(param.normPsi + param.normPhi)>=1
    disp('error choice step-sizes')
end


x = param.x0 ;
if ~isfield(param,'v')
    v = 0 * Psit(x) ;
else
    v = param.v ;
end
if ~isfield(param,'u')
    u = 0 * y ;
else
    u = param.u ;
end

Phix = Phi*x ;
Psitx = Psit(x) ;
l2norm = norm(Phix - y) ;
l1norm = sum(abs(weights.*Psitx)) ;
disp('Initialization - Primal dual algorithm')
disp(['l2 norm           = ',num2str(l2norm)])
disp(['     vs. l2 bound = ',num2str(param.epsilon)])
disp(['l1 norm           = ',num2str(l1norm)])
disp('----------------------------------')


%% proximity operators

sc = @(z, radius) z * min(radius/norm(z(:)), 1);
soft = @(z, T) sign(z) .* max(abs(z)-T, 0);

%% main iterations

for it = 1:param.NbIt
    vold = v ;
    uold = u ;
    xold = x ;
    
    % Dual updates
    
    % regularization
    v_ = v + sigma * Psitx ;
    v = v_ - sigma * soft( v_/sigma, param.lambda * sigma^(-1) *weights) ;
    
    % data fid
    u_ = u + sigma * Phix ;
    u = u_ - sigma * ( sc( u_/sigma - y, param.epsilon) + y ) ;
    
    
    % Primal update
    x = x - tau * (Psi(2*v-vold) + Phi'*(2*u-uold)) ;
    x(x<0) = 0 ;
    
    Phix = Phi*x ;
    l2norm = norm(Phix - y) ;
    Psitx = Psit(x) ;
    l1norm_old = l1norm ;
    l1norm = sum(abs(weights.*Psitx)) ;
    
    condl1 = abs( l1norm - l1norm_old ) / l1norm ;
    condnorm = norm(x(:)-xold(:))/norm(x(:)) ;
    
    if mod(it,param.display) ==0
    disp(['it = ', num2str(it)])
    disp(['l2 norm                 = ',num2str(l2norm)])
    disp(['     vs. l2 bound       = ',num2str(param.epsilon)])
    disp(['     vs. stop l2 bound  = ',num2str((1+param.stopbound) * param.epsilon)])
    disp(['l1 norm                 = ',num2str(l1norm)])
    disp(['cond l1 norm            = ',num2str(condl1)])
    disp(['     stop l1 norm       = ',num2str(param.stopcrit)])
    disp(['cond norm iterates      = ',num2str(condnorm)])
    disp(['     stop norm iterates = ',num2str(param.stopnorm)])
    disp('----------------------------------')
   
    figure(100)
    imagesc(reshape(x,param.Ny,param.Nx)), axis image; colormap gray, colorbar
    xlabel(['it = ',num2str(it)])
    pause(0.1)
    end
    
    if it>10 ...
            &&  condl1 < param.stopcrit ...
            &&  l2norm < (1+param.stopbound) * param.epsilon ...
            &&  condnorm < param.stopnorm
        disp(['stopping criterion reached, it ', num2str(it)])
    disp(['l2 norm                = ',num2str(l2norm)])
    disp(['     vs. l2 bound      = ',num2str(param.epsilon)])
    disp(['     vs. stop l2 bound = ',num2str((1+param.stopbound) * param.epsilon)])
    disp(['l1 norm                 = ',num2str(l1norm)])
    disp(['cond l1 norm            = ',num2str(condl1)])
    disp(['     stop l1 norm       = ',num2str(param.stopcrit)])
    disp(['cond norm iterates      = ',num2str(condnorm)])
    disp(['     stop norm iterates = ',num2str(param.stopnorm)])
    disp('----------------------------------')
        break
    end
    
end

    figure(100)
    imagesc(reshape(x,param.Ny,param.Nx)), axis image; colormap gray, colorbar
    xlabel(['it = ',num2str(it)])
    pause(0.1)


end