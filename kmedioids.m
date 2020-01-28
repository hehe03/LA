function [inds,cidx] = kmedioids(D,k)

 %rng('default'); %fix rand-20190902

% [inds,cidx] = kmedioids(D,k)
%
% Performs k-mediods clustering; only requires a distance matrix D and
% number of clusters k.  Finds cluster assignments "inds" to minimize the
% following cost function: 

% sum(D(inds==i,inds==i),2), summed over i=1:k 
    
% Determining cluster assignments and cluster centers are both done in an
% efficient, vectorized way.  Cluster assignment is O(nk) and cluster
% centering is O(k*(max cluster size)^2)
%
% INPUTS
% D: nxn all-pairs distance matrix
% k: number of clusters
%
% OUTPUTS
% inds: nx1 vector of assignments of each sample to a cluster id
% cidx: kx1 vector of sample indices which make up the cluster centers
%
% DEMO
% Run with no arguments for demo with 2d points sampled from 3 gaussians,
% using the gmdistribution function from the stats toolbox

% Written by Ben Sapp, September 2010
% benjamin.sapp@gmail.com

if nargin == 0
    demo();
    return;
end

n = size(D,1);

% randomly assign centers:
cidx = randperm(n);
cidx = sort(cidx(1:k));


iter = 0;
while 1
    inds = assign_pts_to_clusters(D,cidx);
    [cidx,energy_next] = update_centers(D,inds,k);
    
    if iter>0 && energy_next == energy
        break;
    end
    energy = energy_next;
    
    fprintf('iter: %04d, energy: %.02f\n',iter,energy)
    iter = iter+1;
end

function inds = assign_pts_to_clusters(D,cidx)
S  = D(cidx,:);
[vals,inds] = min(S,[],1);

function [cidx,energy] = update_centers(D,inds,k,pts)
energy = nan(k,1);
for i=1:k
   indsi = find(inds==i);
   [energy(i),minind] = min(sum(D(indsi,indsi),2));
   cidx(i) = indsi(minind);
end
energy = sum(energy);

function demo()
% problem params
k = 3;
n = 2000;
MU = [1 2;-1 -2; 3 0];
SIGMA = cat(3,[2 0;0 .5],[1 0;0 1], eye(2));
p = ones(1,3)/3;
obj = gmdistribution(MU,SIGMA,p);
pts = random(obj,n)';

%form all-pairs distance matrix in an efficient way
X = pts';
temp = sum(X.^2,2);
X=sqrt(2)*X;
D=-X*X';
D=bsxfun(@plus,D,temp);
D=bsxfun(@plus,D,temp');

%run kmedioids
[inds,cidx] = kmedioids(D,k);

%display
clf, hold on, axis square
c = lines(k);
for i=1:k
    ptsi = pts(:,inds==i);
    ctrpt = pts(:,cidx(i));
    plot(ptsi(1,:),ptsi(2,:),'.','color',c(i,:))
    plot(ctrpt(1),ctrpt(2),'kx','markersize',22,'linewidth',6)    
end