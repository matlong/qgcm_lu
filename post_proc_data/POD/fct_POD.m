function [lambda, umodes, vmodes, alpha, um, vm] = fct_POD(u, v)
%% function [lambda, umodes, vmodes, alpha, um, vm] = fct_POD(u, v)
% Snapshot POD procedure of scalar or velocity fluctuations at one layer.
%
% Inputs:
%        u: (1D) scalar of size (nxp,nyp,nt)
%           (2D) U-velocity of size (nxp,nyp-1,nt)
%        v:      V-velocity of size (nxp-1,nyp,nt)
%        where nxp and nyp are p-grid size.
% Outputs:
%        lambda: eigenvalues of (temporal) correlation matrix
%        umodes and vmodes: spatial modes of u and v
%        (Optional) alpha: temporal modes
%        (Optional) um: time-mean of u
%        (Optional) vm: time-mean of v
%
% Written by Long Li 2020-08-10.
%

[nux,nuy,nt] = size(u);
nu = nux*nuy;
if nargin>1
    [nvx,nvy] = size(v(:,:,1));
    nv = nvx*nvy;
end

% Fluctuations of flow
um = mean(u,3);
uf = u - um;
if nargin>1
    vm = mean(v,3);
    vf = v - vm;
end

% Temporal correlation matrix
C = inner_prod(uf); % L2 inner product
if nargin>1
    C = C + inner_prod(vf);
end
C = C./nt; % To have orthonormal spatial modes 

% Solve eigen problem
[V, D] = eig(C);
[lambda, ind_sort] = sort(diag(D), 'descend');
alpha = V(:,ind_sort)';
clear C D V

% Scale temporal modes
for i = 1:nt
   alpha(i,:) = alpha(i,:).*sqrt(nt*max(lambda(i), 0.0));
end

% Build spatial modes
U = reshape(uf, [nu,nt]);
umodes = 0.*u;
if nargin>1
    V = reshape(vf, [nv,nt]);
    vmodes = 0.*v;
end

for i = 1:nt
   umodes(:,:,i) = reshape((U*alpha(i,:)')./(nt*lambda(i)), [nux,nuy]);
   if nargin>1
       vmodes(:,:,i) = reshape((V*alpha(i,:)')./(nt*lambda(i)), [nvx,nvy]);
   end
end

end