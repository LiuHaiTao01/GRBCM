% demo of aggregation GPs for a 1D toy example
clear all
rng(100);

% generate data
n = 500 ; sn = 0.5; nt = 500;
f = @(x) 5*x.^2.*sin(12*x) + (x.^3-0.5).*sin(3*x-0.5) + 4*cos(2*x) ;
x = linspace(0,1,n)';  y = f(x)+sn*randn(n,1);          % training data
xt = linspace(-0.2,1.2,nt)'; yt = f(xt)+sn*randn(nt,1); % test data

%-----------------------------------------------------
%---------------------Aggregation GP------------------
%-----------------------------------------------------
% model parameters
sf2 = 1 ; ell = 1 ; sn2 = 0.1 ; 
partitionCriterion = 'random' ; % 'random', 'kmeans'

% train
opts.Xnorm = 'Y' ; opts.Ynorm = 'Y' ;
opts.Ms = n/50 ; opts.partitionCriterion = partitionCriterion ;
opts.ell = ell ; opts.sf2 = sf2 ; opts.sn2 = sn2 ;
opts.meanfunc = []; opts.covfunc = @covSEard; opts.likfunc = @likGauss; opts.inffunc = @infGaussLik ;
opts.numOptFC = 25 ;
[models,t_dGP_train] = aggregation_train(x,y,opts) ;

% predict
criterion = 'PoE' ; % PoE, GPoE, BCM, RBCM, GRBCM, NPAE
[mu_dGP,s2_dGP,t_dGP_predict] = aggregation_predict(xt,models,criterion) ;


%-----------------------------------------------------
%---------------------Full GP-------------------------
%-----------------------------------------------------
% normalization if required for full GP
if strcmp(models{1}.optSet.Xnorm,'Y')   
    x_norm  = (x - repmat(models{1}.X_mean,n,1)) ./ (repmat(models{1}.X_std,n,1)) ;
    xt_norm = (xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
else
    x_norm = x ; xt_norm = xt ; 
end
if strcmp(models{1}.optSet.Ynorm,'Y')  
    y_norm = (y - repmat(models{1}.Y_mean,n,1)) ./ (repmat(models{1}.Y_std,n,1)) ;
else 
    y_norm = y ;
end

numOptFcn = opts.numOptFC ;
cov = opts.covfunc; mean = opts.meanfunc; lik = opts.likfunc; inf = opts.inffunc;
hyp.cov = log([ell;sf2]); hyp.lik = log(sn2); hyp.mean = [];
hyp_opt = minimize(hyp,@gp,numOptFcn,inf,mean,cov,lik,x_norm,y_norm);
[mu_fullGP,s2_fullGP] = gp(hyp_opt,inf,mean,cov,lik,x_norm,y_norm,xt_norm); 
if strcmp(models{1}.optSet.Ynorm,'Y')
    mu_fullGP = mu_fullGP.*repmat(models{1}.Y_std,nt,1) + repmat(models{1}.Y_mean,nt,1) ;
    s2_fullGP = (models{1}.Y_std)^2*s2_fullGP ;
end

% plot
figure('position',[142    59   843   557])
f = [mu_fullGP+2*sqrt(s2_fullGP); flipdim(mu_fullGP-2*sqrt(s2_fullGP),1)];
fill([xt; flipdim(xt,1)],f,[127,124,119]/256,'facealpha',0.5,'EdgeColor',[127,124,119]/256,'edgealpha',0); hold on
plot(x,y,'+','LineWidth',1.5,'markersize',10,'color',[0,113,188]/256); hold on ;
p1 = plot(xt,mu_fullGP,'k','LineWidth',3); hold on ;
p2 = plot(xt,mu_dGP,'LineWidth',3,'color','r'); hold on ;
plot(xt,mu_dGP+2*sqrt(s2_dGP),'-','LineWidth',2,'color','r'), plot(xt,mu_dGP-2*sqrt(s2_dGP),'-','LineWidth',2,'color','r'); hold on;
h = legend([p1,p2],'full GP',criterion);
grid on;
set(h,'fontsize',16)
xlabel('x') ; ylabel('y') ;
set(gca,'fontsize',16)
set(gcf,'color','w')