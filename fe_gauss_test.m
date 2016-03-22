function [x allFE geweke] = fe_gauss_test(q_mu, q_sig)

% METROPOLIS-HASTINGS BAYESIAN POSTERIOR
rand('seed',12345)

% PRIOR OVER SCALE PARAMETERS
B = 1;

% DEFINE LIKELIHOOD
%likelihood = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y))','y','A','B');
likelihood = inline('normpdf(x,5,1)','x');

% DEFINE PRIOR OVER SHAPE PARAMETERS
prior = inline('normpdf(x,0,1)','x');

% DEFINE THE POSTERIOR
%p = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y)).*sin(pi*A).^2','y','A','B');

p = inline('normpdf(x,5,1).*normpdf(x,0,1)','x');

% INITIALIZE THE METROPOLIS-HASTINGS SAMPLER
% DEFINE PROPOSAL DENSITY
q = inline('normpdf(x,mu,sig)','x','mu','sig');

% SOME CONSTANTS
nSamples = 100000;
N = nSamples;
burnIn = 500;
minn = 0.1; maxx = 5;

% INTIIALZE SAMPLER
x = zeros(1 ,nSamples);
probTrace = zeros(1,nSamples); 
x(1) = q_mu;
t = 1;
be = 1;
acc = 0;
 
allFE = zeros(1,nSamples);

% RUN METROPOLIS-HASTINGS SAMPLER
while t < nSamples
    t = t+1;
 
    % SAMPLE FROM PROPOSAL
    xStar = normrnd(x(t-1),q_sig);

%     feStar = sum(q([x(1:t-1) xStar],x(t-1),q_sig).*log(likelihood([x(1:t-1) xStar]))) - ...
%         (1/be)*sum(q([x(1:t-1) xStar],x(t-1),q_sig).*...
%         log(q([x(1:t-1) xStar],x(t-1),q_sig)./prior([x(1:t-1) xStar])));
%     feStar = feStar/t;

    feStar = sum(q([x(1:t-1) xStar],x(t-1),q_sig).*log(p([x(1:t-1) xStar]))) - ...
        sum(q([x(1:t-1) xStar],x(t-1),q_sig).*log(q([x(1:t-1) xStar],x(t-1),q_sig)));
    feStar = feStar/t;
    
    
    alpha = min([1, p(xStar)/p(x(t-1))]);
      
    % ACCEPT OR REJECT?
    u = rand;
    if u <= alpha
        x(t) = xStar;
        probTrace(t) = p(xStar);
        acc = acc + 1;
        allFE(t) = feStar;
    else
        x(t) = x(t-1);
        probTrace(t) = probTrace(t-1);
        allFE(t) = allFE(t-1);
    end
end

allFE = allFE(burnIn:end);
%% Convergence checks

% Mixing
accrate = acc/N    % Acceptance rate, should be around 0.234

% Running mean
runmean = cumsum(x(burnIn:end)./1:length(x(burnIn:end)));

% Autocorrelation
t = 1;
nn = 100;
xx = x(1:nn);   xx2 = x(end-nn:end);   % First ans Last nn samples
[r lags]   = xcorr(xx-mean(xx), 'coeff');
[r2 lags2] = xcorr(xx2-mean(xx2), 'coeff');


% Geweke test 
split1 = x(1:round(0.1*N));     split2 = x(round(0.5*N):end);
mean1  = mean(split1);              mean2  = mean(split2) ;  
if abs((mean1-mean2)/mean1) < 0.03   % 3% error
   geweke = 1
else
   geweke = 0
end

% % CALCULATE AND DISPLAY THE TARGET SURFACE
yy = linspace(0,5,100);
target = p(yy);

yy = linspace(0,5,100);
proposal = q(yy,q_mu,q_sig);

%% DISPLAY MARKOV CHAIN
figure;
subplot(211);
stairs(x(1:N),1:N, 'k');
hold on;
hb = plot([0 maxx/2],[burnIn burnIn],'g--','Linewidth',2);
ylabel('t'); xlabel('samples, A');
%set(gca , 'YDir', 'reverse');
title('Markov Chain Path');
legend(hb,'Burnin');
 
% DISPLAY SAMPLES
subplot(212);
nBins = 100;
sampleBins = linspace(minn,maxx,nBins);
counts = hist(x(burnIn:end), sampleBins);
bar(sampleBins, counts/sum(counts), 'k');
xlabel('samples, mu' ); ylabel( 'p(mu | x)' );
title('Samples');
 
% OVERLAY TARGET DISTRIBUTION
hold on;
plot(yy, target/sum(target) , 'm-', 'LineWidth', 2);
plot(yy, proposal/sum(proposal) , 'g-', 'LineWidth', 2);
legend('Sampled Distribution',sprintf('Target Posterior'),sprintf('Inital Proposal'))
axis tight

xlim([0.5 5])

% subplot(313);
% plot(allFE);
% ylim([min(allFE) max(allFE)]);