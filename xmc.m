function xmc
% Boltzmann MCMC

rand('seed',12345);
 
% Step size
delta = 0.3;
nSamples = 100000;
burnIn = 500;
L = 20;

% Define your constants
mu_u = [5]; 
sigma_u = [0.5];
sig_s = 0.5;
mu0 = 0;
sigma0 = 1;
sig_a = 1;

% Likelihood: this is your potential energy function
U = @(s) gauss(s,mu_u,sigma_u);
 
% Gradient of potential energy
dU = @(s) dgauss(s,mu_u, sigma_u);

% Initial
s = zeros(nSamples,1);
s0 = [0];
s(1) = s0;
t = 1;
aStar = 0;
sStar = 0;
be = 1;

a = zeros(nSamples,1);
F = zeros(nSamples,1);

while t < nSamples
    t = t + 1;
 
    % Sample a random action
    aStar = normrnd(a(t-1),sig_a);
    sStar = normrnd(s(t-1),sig_s);
 
%     %% Leapfrog
%     % Take first half step
%     aStar = a0 - delta/2*dU(s(t-1))';
%  
%     % Step forward in state space
%     sStar = s(t-1) + delta*aStar;
% 
%     for jL = 1:L-1
%         % action
%         aStar = aStar - delta*dU(sStar)';
%         % position
%         sStar = sStar + delta*aStar;
%     end
%  
%     % Second half step
%     aStar = aStar - delta/2*dU(sStar)';
       
    % AR 
    FE_star = fe_MC(be,[sStar aStar],[sig_s sig_a],[mu_u 0], [sigma_u 1], mu0, sigma0);
    FE = fe_MC(be,[s(t-1) a(t-1)],[sig_s sig_a],[mu_u 0], [sigma_u 1], mu0, sigma0);
   
    ar = exp(FE_star - FE);
    alpha = min(1,ar);

    u = rand;
    if u < alpha
        s(t) = sStar;
        a(t) = aStar;
        F(t) = FE_star;
    else
        s(t) = s(t-1);
        a(t) = a(t-1);
        F(t) = F(t-1);
    end
end
figure(101);
subplot(3,1,1);
plot(F)
subplot(3,1,2);
hist(s(burnIn:end))
subplot(3,1,3);
hist(a(burnIn:end))
 
keyboard
function y = gauss(x,mu,sig)
% normal gaussian
y = exp(-(x-mu).^2./(2*sig.^2)) ./ (sig*sqrt(2*pi));

function y = dgauss(x,mu, sig)
%first order derivative of Gaussian wrt x
y = -(x-mu) * gauss(x,mu,sig) / sig^2;