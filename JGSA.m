function [Xs, Xt, A, Att] = JGSA(Xs, Xt, Ys, Yt0, Yt, options)

%% 2019.09.1: change classifier from fitcknn to fitcsvm

if nargin<6; options=[]; end

if ~isfield(options,'alpha');    options.alpha=1; end % the parameter for subspace divergence ||A-B||
if ~isfield(options,'mu');        options.mu=1; end % the parameter for target variance
if ~isfield(options,'k');        options.k=20; end % #subspace bases
if ~isfield(options,'beta');        options.beta=0.01; end % the parameter for P and Q (source discriminaiton) 
if ~isfield(options,'gamma');        options.gamma=2; end
if ~isfield(options,'ker');    options.ker='primal'; end
if ~isfield(options,'T');    options.T=1; end

% Joint Geometrical and Statistical Alignment for Visual Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
% Jing Zhang, Wanqing Li, Philip Ogunbona.

alpha = options.alpha;
mu = options.mu;
beta = options.beta;
gamma = options.gamma;
ker = options.ker;
k = options.k;
T = options.T;

m = size(Xs,1);
ns = size(Xs,2);
nt = size(Xt,2);

class = unique(Ys);
C = length(class);
if strcmp(ker,'primal')
    
    %--------------------------------------------------------------------------
    % compute LDA
    dim = size(Xs,1);
    meanTotal = mean(Xs,2);

    Sw = zeros(dim, dim);
    Sb = zeros(dim, dim);
    for i=1:C
        Xi = Xs(:,find(Ys==class(i)));
        meanClass = mean(Xi,2);
        Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
        Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
        Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
    end
    P = zeros(2*m,2*m);
    P(1:m,1:m) = Sb;
    Q = Sw;

    for t = 1:T
        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

        Ts = Xs*Ms*Xs';
        Tt = Xt*Mt*Xt';
        Tst = Xs*Mst*Xt';
        Tts = Xt*Mts*Xs';
        
        % Construct centering matrix
        Ht = eye(nt)-1/(nt)*ones(nt,nt);
        
        X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
        H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];

        Smax = mu*X*H*X'+beta*P;
        Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
                Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');
        A = W(1:m, :);
        Att = W(m+1:end, :);

        Zs = A'*Xs;
        Zt = Att'*Xt;
        
        if T>1
            model = fitcsvm(Zs',Ys);  
            Yt0 =predict(model,Zt');
            acc = length(find(Yt0==Yt))/length(Yt); 
            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
        end
    end
else
    
    Xst = [Xs, Xt];   
    nst = size(Xst,2); 
    [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);
   %--------------------------------------------------------------------------
    % compute LDA
    dim = size(Ks,2);
    C = length(class);
    meanTotal = mean(Ks,1);

    Sw = zeros(dim, dim);
    Sb = zeros(dim, dim);
    for i=1:C
        Xi = Ks(find(Ys==class(i)),:);
        meanClass = mean(Xi,1);
        Hi = eye(size(Xi,1))-1/(size(Xi,1))*ones(size(Xi,1),size(Xi,1));
        Sw = Sw + Xi'*Hi*Xi; % calculate within-class scatter
        Sb = Sb + size(Xi,1)*(meanClass-meanTotal)'*(meanClass-meanTotal); % calculate between-class scatter
    end
    P = zeros(2*nst,2*nst);
    P(1:nst,1:nst) = Sb;
    Q = Sw;        

    for t = 1:T

        % Construct MMD matrix
        [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);
        
        Ts = Ks'*Ms*Ks;
        Tt = Kt'*Mt*Kt;
        Tst = Ks'*Mst*Kt;
        Tts = Kt'*Mts*Ks;

        K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
        Smax =  mu*K'*K+beta*P;
        
        Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
                Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
        [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), k, 'LM');
        W = real(W);

        A = W(1:nst, :);
        Att = W(nst+1:end, :);

        Zs = A'*Ks';
        Zt = Att'*Kt';

        if T>1
            model = fitcsvm(Zs',Ys);  
            Yt0 =predict(model,Zt');  
            acc = length(find(Yt0==Yt))/length(Yt); 
            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
        end
    end
end
    
Xs = Zs;
Xt = Zt;
end


%% -------------------------------------------------
function K = km_kernel(X1,X2,ktype,kpar)
% KM_KERNEL calculates the kernel matrix between two data sets.
% Input:	- X1, X2: data matrices in row format (data as rows)
%			- ktype: string representing kernel type
%			- kpar: vector containing the kernel parameters
% Output:	- K: kernel matrix
% USAGE: K = km_kernel(X1,X2,ktype,kpar)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2012.
%
% This file is part of the Kernel Methods Toolbox for MATLAB.
% https://github.com/steven2358/kmbox

switch ktype
	case 'gauss'	% Gaussian kernel
		sgm = kpar;	% kernel width
		
		dim1 = size(X1,1);
		dim2 = size(X2,1);
		
		norms1 = sum(X1.^2,2);
		norms2 = sum(X2.^2,2);
		
		mat1 = repmat(norms1,1,dim2);
		mat2 = repmat(norms2',dim1,1);
		
		distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
        sgm = sgm / mean(mean(distmat)); % added by jing 24/09/2016, median-distance
		K = exp(-distmat/(2*sgm^2));
		
	case 'gauss-diag'	% only diagonal of Gaussian kernel
		sgm = kpar;	% kernel width
		K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
		
	case 'poly'	% polynomial kernel
% 		p = kpar(1);	% polynome order
% 		c = kpar(2);	% additive constant
        p = kpar; % jing
        c = 1; % jing
		
		K = (X1*X2' + c).^p;
		
	case 'linear' % linear kernel
		K = X1*X2';
		
	otherwise	% default case
		error ('unknown kernel type')
end
end

%% -------------------------------------------------
function [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma)

Xst = [Xs, Xt];   
ns = size(Xs,2);
nt = size(Xt,2);
nst = size(Xst,2); 
Kst0 = km_kernel(Xst',Xst',ker,gamma);
Ks0 = km_kernel(Xs',Xst',ker,gamma);
Kt0 = km_kernel(Xt',Xst',ker,gamma);

oneNst = ones(nst,nst)/nst;
oneN=ones(ns,nst)/nst;
oneMtrN=ones(nt,nst)/nst;
Ks=Ks0-oneN*Kst0-Ks0*oneNst+oneN*Kst0*oneNst;
Kt=Kt0-oneMtrN*Kst0-Kt0*oneNst+oneMtrN*Kst0*oneNst;
Kst=Kst0-oneNst*Kst0-Kst0*oneNst+oneNst*Kst0*oneNst;
end

%% -------------------------------------------------
function [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C)
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
es = 1/ns*ones(ns,1);
et = -1/nt*ones(nt,1);

M = e*e'*C;
Ms = es*es'*C;
Mt = et*et'*C;
Mst = es*et'*C;
Mts = et*es'*C;
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        es = zeros(ns,1);
        et = zeros(nt,1);
        es(Ys==c) = 1/length(find(Ys==c));
        et(Yt0==c) = -1/length(find(Yt0==c));
        es(isinf(es)) = 0;
        et(isinf(et)) = 0;
        Ms = Ms + es*es';
        Mt = Mt + et*et';
        Mst = Mst + es*et';
        Mts = Mts + et*es';
    end
end

Ms = Ms/norm(M,'fro');
Mt = Mt/norm(M,'fro');
Mst = Mst/norm(M,'fro');
Mts = Mts/norm(M,'fro');
end