clear

% METROPOLIS-HASTINGS BAYESIAN POSTERIOR
rand('seed',12345)

% PRIOR OVER SCALE PARAMETERS
B = 1;

% DEFINE LIKELIHOOD
likelihood = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y))','y','A','B');

% DEFINE PRIOR OVER SHAPE PARAMETERS
prior = inline('sin(pi*A).^2','A');

% DEFINE THE POSTERIOR
p = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y)).*sin(pi*A).^2','y','A','B');

% INITIALIZE THE METROPOLIS-HASTINGS SAMPLER
% DEFINE PROPOSAL DENSITY
q = inline('exppdf(x,mu)','x','mu');

% MEAN FOR PROPOSAL DENSITY
mu = 5;
y = 1.5;

% SOME CONSTANTS
nSamples = 10000;
N = nSamples;
burnIn = 1000;
minn = 0.1; maxx = 5;
be = 100000;

% INTIIALZE SAMPLER
x = zeros(1 ,nSamples);
probTrace = zeros(1,nSamples);
x(1) = mu;
t = 1;

acc = 0;
 
% RUN METROPOLIS-HASTINGS SAMPLER
while t < nSamples
    t = t+1;
 
    % SAMPLE FROM PROPOSAL
    xStar = exprnd(mu);
 
    % CORRECTION FACTOR
    c = q(x(t-1),mu)/q(xStar,mu);
 
    % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
    
    feStar = sum(q(xStar,mu).*likelihood(y,xStar,B)) - ...
        (1/be)*sum(q(xStar,mu).*log(q(xStar,mu)./prior(xStar)));
    fe = sum(q(x(t-1),mu).*likelihood(y,x(t-1),B)) - ...
        (1/be)*sum(q(x(t-1),mu).*log(q(x(t-1),mu)./prior(x(t-1))));
    alpha = min(1,exp(-feStar+fe));
    
    % alpha = min([1, p(y,xStar,B)/p(y,x(t-1),B)*c]);
 
    % ACCEPT OR REJECT?
    u = rand;
    if u <= alpha
        x(t) = xStar;
        probTrace(t) = p(y,xStar,B);
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
accrate = acc/N    % Acceptance rate

% Running mean
runmean = cumsum(x)./1:length(x);
figure(104);
plot(runmean);

% Autocorrelation
t = 1;
% for nn = 100:100:1000
%     xx = x(1:nn);   xx2 = x(end-nn:end);   % First ans Last nn samples
%     [r{t} lags]   = xcorr(xx-mean(xx), 'coeff');
%     [r2{t} lags2] = xcorr(xx2-mean(xx2), 'coeff');
% end
nn = 100
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

% CALCULATE AND DISPLAY THE POSTERIOR SURFACE
yy = linspace(0,10,100);
AA = linspace(0.1,5,100);
likeSurf = zeros(numel(yy),numel(AA));
for iA = 1:numel(AA)
    likeSurf(:,iA)=likelihood(yy(:),AA(iA),B);
end

postSurf = zeros(size(likeSurf));
for iA = 1:numel(AA)
    postSurf(:,iA)=p(yy(:),AA(iA),B);
end
 
% SAMPLE FROM p(A | y = 1.5)
y = 1.5;
target = postSurf(16,:);

%% DISPLAY MARKOV CHAIN
figure(101);
subplot(211);
stairs(x(1:N),1:N, 'k');
hold on;
hb = plot([0 maxx/2],[burnIn burnIn],'g--','Linewidth',2);
ylabel('t'); xlabel('samples, A');
set(gca , 'YDir', 'reverse');
%ylim([0 t])
%axis tight;
%xlim([0 maxx]);
title('Markov Chain Path');
legend(hb,'Burnin');
 
% DISPLAY SAMPLES
subplot(212);
nBins = 100;
sampleBins = linspace(minn,maxx,nBins);
counts = hist(x(burnIn:end), sampleBins);
bar(sampleBins, counts/sum(counts), 'k');
xlabel('samples, A' ); ylabel( 'p(A | y)' );
title('Samples');
%xlim([0 10])
 
% OVERLAY TARGET DISTRIBUTION
hold on;
plot(AA, target/sum(target) , 'm-', 'LineWidth', 2);
legend('Sampled Distribution',sprintf('Target Posterior'))
axis tight

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