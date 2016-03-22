function FE = fe_MC(be,mu, sigma,mu_u, sigma_u, mu0, sigma0)
% This calculates negative delta free energy for a given posterior, prior and
% utility function

X = mvnrnd(mu, sigma,1);

FE = sum(mvnpdf(X,mu,sigma).*mvnpdf(X,mu_u,sigma_u) - ...
    (1/be)*sum(mvnpdf(X,mu,sigma).*log(mvnpdf(X,mu,sigma)./normpdf(X(:,1),mu0,sigma0))));
FE = sum(FE); % maximise negative free energy
