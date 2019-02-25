%% ************************************************************************
%  ************************************************************************
%  Compressive optical imaging with a photonic lantern
%  version 0.0
%  November 2018
%
%  Author: Audrey Repetti
%  Contact: arepetti@hw.ac.uk
%
%  For details on the method, refer to the article
%  Compressive optical imaging with a photonic lantern
%  ************************************************************************
%  ************************************************************************

%%
clear all
close all
clc


%% 

% ---------------------------------------------------------
% load data
load('cross1089.mat')
% ---------------------------------------------------------
% image_choice : name of the image of interest
% nb_pat       : number of considered patterns
% Nx, Ny       : dimension of the image of interest
% Phi          : measurement operator
% ydata        : observed data
% xtrue        : true image - for visual comparison only
% ---------------------------------------------------------
% MODEL    : ydata = Phi * xtrue
% OBJECTIVE: Find an estimate xrec of xtrue from ydata
% ---------------------------------------------------------

% ---------------------------------------------------------
% CREATE NOISY DATA
% input SNR - level of Gaussian noise to add to the simulated measurements
isnr = Inf ; % choose Inf to not add noise
if isnr < Inf
    sigma_noise = norm(ydata(:))/sqrt(numel(ydata)) * 10^(-isnr/20) ;
    ydata = ydata + sigma_noise * randn(size(ydata)) ;
end
% ---------------------------------------------------------

% ---------------------------------------------------------
% compute norm operator (can take few seconds...)
normPhi_ = pow_method(@(x) Phi * x , @(y) Phi'*y, [Nx*Ny, 1]) ;
% ---------------------------------------------------------



%% Algorithm parameters


% Global parameters
param.Nx = Nx;
param.Ny = Ny;
param.normPhi = normPhi_ ;
param.epsilon = 50 ; 
[param.Psi, param.Psit] = SARA_sparse_operator(rand([param.Nx param.Ny]), 2) ;
param.normPsi = 1 ;

% algorithm parameters
param.lambda = 1e-3 ; % parameter acting on convergence speed
param.stopcrit = 1e-3 ;
param.stopnorm = 1e-3 ;
param.stopbound = 1e-2 ; 
param.NbIt = 2000;
param.display = 100;

% re-weighting parameters
NbRW = 3 ; % number of re-weightings
sigma = 0.1 ;



%% Reconstruction method


% First estimate without weights
Xrec = cell(NbRW+1,1) ;
param.x0 = zeros(param.Nx*param.Ny,1) ; 
[Xrec{1},param.u,param.v] = PD_ghost_SARA(ydata, Phi, param) ;
pause(0.1)

% Re-weighting procedure
for iterW = 1:NbRW
    disp(' ')
    disp('**********************************')
    disp(['Re-weighting iteration: ',num2str(iterW)])
    temp=param.Psit(reshape(Xrec{iterW},param.Ny,param.Nx));
    delta=std(temp(:));
    sigma_n = sigma*sqrt(numel(ydata)/(param.Ny*param.Nx*1));
    delta=max(sigma_n/10,delta);
    
    
    % Weights
    weights=abs(param.Psit(reshape(Xrec{iterW},param.Ny,param.Nx)));
    weights=delta./(delta+weights);
    param.weights = weights ;
    
    param.x0 = Xrec{iterW} ;
    [Xrec{iterW+1},param.u,param.v] = PD_ghost_SARA(ydata, Phi, param) ;
end

figure, 
imagesc(reshape(Xrec{iterW+1},Nx,Ny)), axis image, colormap gray; axis off; colorbar('FontWeight','bold','FontSize',12)
xlabel(['constrained pb, eps=',num2str(param.epsilon),' -- RW ',num2str(NbRW)],'FontWeight','bold','FontSize',14)
pause(0.1)



