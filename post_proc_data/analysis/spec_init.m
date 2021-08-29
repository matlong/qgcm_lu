function wave = spec_init (nx2, dx2)
%% function wave = spec_init (nx2, dx2)
% Create Fourier grid of 2D wavenumber and 1D wavenumber.
%
% Inputs: 
%        nx2 = [nx ny] is 1D array of dimensions;
%        dx2 = [dx dy] is 1D array of grid spacings.
%
% Ouputs:
%        wave.k2 is 1D array of horizontal wavenumbers;
%        wave.k1 is 1D array of isotropic wavenumbers;
%        wave.kfac is a scalar to rescale the power spectral density.
%
% Written by Long Li 2020-08-10.
%

wave = [];

% 2D Wavevectors
dk2 = 2*pi./(nx2.*dx2); % wave-step
kx = dk2(1)*([0:nx2(1)/2,1-nx2(1)/2:-1]);
ky = dk2(2)*([0:nx2(2)/2,1-nx2(2)/2:-1]);
[kx,ky] = ndgrid(kx,ky);
k2 = sqrt(kx.^2+ky.^2);  % horizontal wavenumbers
wave.k2 = k2(:); clear kx ky

% 1D isotropic wavenumbers
nk1 = ceil(sqrt(2)*max(nx2./2)); % size of rings
dk1 = max(dk2); % step
wave.k1 = (1:nk1).*dk1;

% Factor for energy spectrum
wave.kfac = prod(dx2)*min(dk2)/(8*pi^2)/prod(nx2);

end