function c = inner_prod(u)
%% function c = inner_prod(u)
% Compute L2 inner product of u, where u is of size (nx,ny,nt).
%
% Written by Long Li 2020-08-10.
%

% Get domain size
[nx,ny,nt] = size(u);

if mod(nx,2)==0 % even nb. of cells
    wefac = 1; % weight for western and eastern boundaries
    afac = 1/nx; % weight of area 
else
    wefac = 0.5;
    afac = 1/(nx-1);
end

if mod(ny,2)==0
    snfac = 1; % weight for southern and northern boundaries
    afac = afac/ny;
else
    snfac = 0.5;
    afac = afac/(ny-1);
end
cfac = snfac*wefac; % weight for corner points

% Integration over inner points
np = (nx-2)*(ny-2); % number of points
utmp = reshape(u(2:nx-1,2:ny-1,:), [np,nt]);
c = utmp'*utmp; % of size (nt,nt)
clear utmp

% Integration over southern boundary
np = nx-2;
utmp = reshape(u(2:nx-1,1,:), [np,nt]);
c = c + snfac.*(utmp'*utmp); clear utmp

% Integration over northern boundary
utmp = reshape(u(2:nx-1,ny,:), [np,nt]);
c = c + snfac.*(utmp'*utmp); clear utmp

% Integration over western boundary
np = ny-2;
utmp = reshape(u(1,2:ny-1,:), [np,nt]);
c = c + wefac.*(utmp'*utmp); clear utmp

% Integration over eastern boundary
utmp = reshape(u(nx,2:ny-1,:), [np,nt]);
c = c + wefac.*(utmp'*utmp); clear utmp

% Integration over southern corner points
np = 2;
utmp = reshape(u([1,nx],1,:), [np,nt]);
c = c + cfac.*(utmp'*utmp); clear utmp

% Integration over northern corner points
utmp = reshape(u(1,[1,ny],:), [np,nt]);
c = c + cfac.*(utmp'*utmp); clear utmp

% Normalize by area-size
c = c.*afac;

end