% True posterior mean = 2.5
% True posterior variance = 0.5

clear; close all;

% Scale proposal mean and variance 
qs = [2.5 5; 2.5 1; 2.5 0.5; 2.5 0.25];
for i=1:size(qs,1)
    [x F geweke(i)] = fe_gauss_test(qs(i,1),qs(i,2));
    allX(:,i) = x';
    allF(:,i) = F';
end

f_final = mean(allF(end-100:end,:),1);
f_initial = mean(allF(1:100,:),1);
fdiff = f_final-f_initial;
