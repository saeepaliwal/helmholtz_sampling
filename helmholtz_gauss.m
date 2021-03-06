clear; close all;

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
q_sig = 2;
q = inline('normpdf(x,0,sig)','x','sig');

% MEAN FOR PROPOSAL DENSITY
mu = 0;
y = 1.5;

% SOME CONSTANTS
xmc = 1;
if ~xmc
    % Normal mcmc
    nSamples = 5000;
    N = nSamples;
    burnIn = 500;
    minn = 0.1; maxx = 5;
else
    % XMC
    nSamples = 5000;
    N = nSamples;
    burnIn = 500;
    minn = 0.1; maxx = 5;
end

% INTIIALZE SAMPLER
x = zeros(1 ,nSamples);
probTrace = zeros(1,nSamples);
x(1) = mu;
t = 1;
be = 1;
acc = 0;
 
% RUN METROPOLIS-HASTINGS SAMPLER
while t < nSamples
    t = t+1;
    
    % SAMPLE FROM PROPOSAL
    xStar = normrnd(x(t-1),q_sig);
 
    % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
    %alpha = min([1, p(xStar)/p(x(t-1))]);
    
%     % FE AR CRITERION
%     feStar = sum(q(xStar,q_sig).*likelihood(xStar)) - ...
%         (1/be)*sum(q(xStar,q_sig).*log(q(xStar,q_sig)./prior(xStar)));
%     fe = sum(q(x(t-1),q_sig).*likelihood(x(t-1))) - ...
%         (1/be)*sum(q(x(t-1),q_sig).*log(q(x(t-1),q_sig)./prior(x(t-1))));
    
    feStar = sum(q([x(1:t-1) xStar],q_sig).*likelihood([x(1:t-1) xStar])) - ...
        (1/be)*sum(q([x(1:t-1) xStar],q_sig).*...
        log(q([x(1:t-1) xStar],q_sig)./prior([x(1:t-1) xStar])));
    feStar = feStar/t;
    
    fe = sum(q(x(1:t-1),q_sig).*likelihood(x(1:t-1))) - ...
        (1/be)*sum(q(x(1:t-1),q_sig).*...
        log(q(x(1:t-1),q_sig)./prior(x(1:t-1))));
    fe = fe/(t-1);
    %      if feStar>fe
    %          keyboard
    %      end
    %     feStar = be*log(sum(p(x(1:t-1)))/(t-1));
    %     fe = be*log(sum(p([x(1:t-2) xStar]))/(t-1));
    
    %     alpha = min(1,exp(feStar-fe));
    alpha = min([1, p(xStar)/p(x(t-1))]);
    
    % ACCEPT OR REJECT?
    u = rand;
    if u <= alpha
        x(t) = xStar;
        probTrace(t) = p(xStar);
        acc = acc + 1;
    else
        x(t) = x(t-1);
        probTrace(t) = probTrace(t-1);
    end
end

%%
figure(102);
subplot(2,1,1);
hist(x(burnIn:end));
subplot(2,1,2);
plot(probTrace(burnIn:end));


%% Convergence checks

% Mixing
accrate = acc/N    % Acceptance rate, should be around 0.234

% Running mean
runmean = cumsum(x(burnIn:end)./1:length(x(burnIn:end)));
figure(104);
plot(runmean);

% Autocorrelation
t = 1;
nn = 100;
xx = x(1:nn);   xx2 = x(end-nn:end);   % First ans Last nn samples
[r lags]   = xcorr(xx-mean(xx), 'coeff');
[r2 lags2] = xcorr(xx2-mean(xx2), 'coeff');
% figure(102)
% subplot(211);
% plot(r);
% subplot(212);
% plot(r2);

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

%% DISPLAY MARKOV CHAIN
figure(101);
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
legend('Sampled Distribution',sprintf('Target Posterior'))
axis tight
xlim([0.5 5])
% %% EXTRA PLOTS
% % VISUALIZE POSTERIOR SURFACE
% figure(102);
% surf(postSurf); ylabel('y'); xlabel('A'); colormap hot
% % DISPLAY THE PRIOR
% hold on; pA = plot3(1:100,ones(1,numel(AA))*100,prior(AA),'b','linewidth',3)
% % DISPLAY POSTERIOR
% psA = plot3(1:100, ones(1,numel(AA))*16,postSurf(16,:),'m','linewidth',3)
% xlim([0 100]); ylim([0 100]);  axis normal
% set(gca,'XTick',[0,100]); set(gca,'XTickLabel',[0 5]);
% set(gca,'YTick',[0,100]); set(gca,'YTickLabel',[0 10]);
% view(65,25)
% legend([pA,psA],{'p(A)','p(A|y = 1.5)'},'Location','Northeast');
% hold off
% title('p(A|y)');
% 
% % DISPLAY TARGET AND PROPOSAL
% figure(103); 
% hold on;
% th = plot(AA,target,'m','Linewidth',2);
% qh = plot(AA,q(AA,mu),'k','Linewidth',2)
% legend([th,qh],{'Target, p(A)','Proposal, q(A)'});
% xlabel('A');