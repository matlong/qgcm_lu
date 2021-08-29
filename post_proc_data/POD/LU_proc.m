% Pre-process LU variables from the data produced by 'POD.m'.
% 
% Written by Long Li 2020-08-15.
%

clear all; close all; clc;

%% Set parameters
% file_dir is the directory of input ans outputs.
% run_name is the name given for the folder of a particular run within
%             these locations.
% infile and outfile are the name of input and output NetCDF files.

indir = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/POD';
infile = 'oceof80.nc'; 

outdir = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/POD';
outfile = 'ocludat80.nc'; 

% Correlation time (s)
% tau can be approximated by dto_l*(dxo_L/dxo_l)^(1/3).
% dta = tau/3 should divide 14400.
tau = 1440; 
% Ocean:
% 5km - 600s; 10 km - 720s; 20km - 960s; 40km - 1200s; 80km - 1440s; 120km - 1800s;   

% Option for illustration of noise (1/0)
plot_noise = 1;

%% Read data
% Read full EOFs associated with all eigenvalues of covariance,
% the mean correction from residuals and the mean of energy spectrums.
%

disp('Read input data')

file1 = fullfile(indir,infile);

% Get axis and dimensions
xpo = ncread(file1, 'xp'); nxpo = length(xpo);
ypo = ncread(file1, 'yp'); nypo = length(ypo);
xto = ncread(file1, 'xt'); nxto = length(xto);
yto = ncread(file1, 'yt'); nyto = length(yto);
zo  = ncread(file1, 'z');  nlo  = length(zo);

% Get eigenvalues (L2-norm of EOFs)
lambda = ncread(file1,'lambda'); 
idm = 1:size(lambda,1); % Indices of mode

% Plot of RIC
figure(1);
for k = 1:nlo
    plot(idm(1:500), lambda(1:500,k), '-o', 'LineWidth', 2); hold on;
    myleg{k} = strcat('Layer', num2str(k));
end
hold off; legend(myleg); grid minor; title('Spectrum of modes');

figure(2);
for k = 1:nlo
    RIC(:,k) = cumsum(lambda(:,k))/sum(lambda(:,k));
    plot(idm(1:500), RIC(1:500,k), '-o', 'LineWidth', 2); hold on;
    myleg{k} = strcat('Layer', num2str(k));
end
hold off; legend(myleg); grid minor; title('Energy proportion');

disp(' ')

%% Define inhomogeneous noise
% Truncate the spatial modes to construct the span basis of noise.
% Two input parameters 'mod0' and 'nmo' are required, where the first one
% is the index of largest structure of the noise, whereas the latter one
% is the total number of modes used to expand the noise.
% Once the EOFs are truncated, the variance tensor will be deduced.
%

disp('Building EOFs of noise')

% Get noise mean
uco = ncread(file1,'usubs');
vco = ncread(file1,'vsubs');

% Number of fluctuation modes
nmo = input('Please set the number of modes used for noise: ');

% Read spatial modes
umode = ncread(file1, 'umode', [1 1 1 1], [Inf Inf Inf nmo]);
vmode = ncread(file1, 'vmode', [1 1 1 1], [Inf Inf Inf nmo]);   

ueof = zeros(nxpo,nyto,nlo,nmo);
veof = zeros(nxto,nypo,nlo,nmo);
for k = 1:nlo
    for l = 1:nmo
        ueof(:,:,k,l) = umode(:,:,k,l).*sqrt(lambda(l,k));
        veof(:,:,k,l) = vmode(:,:,k,l).*sqrt(lambda(l,k));
    end
    clear tmp
end
idm = idm(2:nmo+1); clear umode vmode

disp('Derive variance tensor of noise')

axxo = tau.*sum(bsxfun(@times, ueof, ueof), 4);
ayyo = tau.*sum(bsxfun(@times, veof, veof), 4);
axyo = zeros(nxpo,nypo,nlo);
axyo(2:nxto,2:nyto,:) = 0.25*tau.*sum(bsxfun(@times, ...
                ueof(2:nxto,2:nyto,:,:) + ueof(2:nxto,1:nyto-1,:,:), ...
                veof(2:nxto,2:nyto,:,:) + veof(1:nxto-1,2:nyto,:,:)), 4);

% Reshape EOFs
nuo = prod(size(ueof(:,:,:,1)));
nvo = prod(size(veof(:,:,:,1)));
npo = nuo + nvo;
eofo = zeros(npo,nmo);
for j=1:nmo
    eofo(1:nuo,j) = reshape(ueof(:,:,:,j), [nuo,1]);
    eofo(nuo+1:npo,j) = reshape(veof(:,:,:,j), [nvo,1]);
end
clear ueof veof

disp(' ')

%% (Optional) Illustration of noises
% Plot snapshot of noises.
%

if plot_noise

    disp('Generate random velocities')
    
    cmap = getPyPlot_cMap('RdBu_r');
    cax = @(x) max(abs(x(:)));
    [xum, yum] = ndgrid(xpo, yto);
    [xvm, yvm] = ndgrid(xto, ypo);
    [xtm, ytm] = ndgrid(xto, yto);
    
    % Karhunen Loève decomposition 
    tmp = eofo * randn([nmo,1]);
    u_POD = reshape(tmp(1:nuo), [nxpo nyto nlo]);
    v_POD = reshape(tmp(nuo+1:end), [nxto nypo nlo]); clear tmp
    
    figure(3);
    for k=1:nlo
        subplot(3,2,1+(k-1)*2);
        utmp = u_POD(:,:,k) + uco(:,:,k);
        pcolor(xum,yum,utmp); shading('interp');
        tmp = cax(u_POD(:,:,k))/2;
        colormap(cmap); colorbar; caxis([-tmp tmp]); clear tmp
        title(sprintf('Zonal velocity of layer %d', k));
        subplot(3,2,2*k);
        vtmp = v_POD(:,:,k) + vco(:,:,k);
        pcolor(xvm,yvm,vtmp); shading('interp');
        tmp = cax(v_POD(:,:,k))/2;
        colormap(cmap); colorbar; caxis([-tmp tmp]); clear tmp
        title(sprintf('Meridional velocity of layer %d', k));
    end
  
    disp(' ')

end

%% Save LU data
% Save {'eofo','uco','vco','axxo','ayyo','axyo'} into 'outfile'.
%

disp('Save NetCDF data');

% Create new netCDF file
tmp = fullfile(outdir,outfile);
if exist(tmp,'file')==2
   delete(tmp);
end
cmode = netcdf.getConstant('NOCLOBBER');
cmode = bitor(cmode,netcdf.getConstant('64BIT_OFFSET'));
ncid = netcdf.create(tmp,cmode); clear tmp

% Define dimensions
mdim = netcdf.defDim(ncid,'mode',nmo);
xpdim = netcdf.defDim(ncid,'xp',nxpo);
ypdim = netcdf.defDim(ncid,'yp',nypo);
zdim = netcdf.defDim(ncid,'z',nlo);
xtdim = netcdf.defDim(ncid,'xt',nxto);
ytdim = netcdf.defDim(ncid,'yt',nyto);
xydim = netcdf.defDim(ncid,'xy',npo);

% Define 1D variables
mid = netcdf.defVar(ncid,'mode','NC_INT',mdim);
xpid = netcdf.defVar(ncid,'xp','NC_DOUBLE',xpdim);
ypid = netcdf.defVar(ncid,'yp','NC_DOUBLE',ypdim);
zid = netcdf.defVar(ncid,'z','NC_DOUBLE',zdim);
xtid = netcdf.defVar(ncid,'xt','NC_DOUBLE',xtdim);
ytid = netcdf.defVar(ncid,'yt','NC_DOUBLE',ytdim);

% Define 2D variables
dims = [xydim mdim]; % EOFs
eofid = netcdf.defVar(ncid,'eofo','NC_DOUBLE',dims); 

% Define 3D variables
dims = [xpdim ytdim zdim]; % mean and variance
ucid = netcdf.defVar(ncid,'uco','NC_DOUBLE',dims);
axxid = netcdf.defVar(ncid,'axxo','NC_DOUBLE',dims);

dims = [xtdim ypdim zdim];
vcid = netcdf.defVar(ncid,'vco','NC_DOUBLE',dims);
ayyid = netcdf.defVar(ncid,'ayyo','NC_DOUBLE',dims); 

dims = [xpdim ypdim zdim];
axyid = netcdf.defVar(ncid,'axyo','NC_DOUBLE',dims); 

% Define attributes
netcdf.putAtt(ncid,mid,'long name','Index of modes');
netcdf.putAtt(ncid,xpid,'units','km');
netcdf.putAtt(ncid,xpid,'long name','X axis (p-grid)');
netcdf.putAtt(ncid,ypid,'units','km');
netcdf.putAtt(ncid,ypid,'long name','Y axis (p-grid)');
netcdf.putAtt(ncid,zid,'units','km');
netcdf.putAtt(ncid,zid,'long name','Layer coord.');
netcdf.putAtt(ncid,xtid,'units','km');
netcdf.putAtt(ncid,xtid,'long name','X axis (T-grid)');
netcdf.putAtt(ncid,ytid,'units','km');
netcdf.putAtt(ncid,ytid,'long name','Y axis (T-grid)');
netcdf.putAtt(ncid,eofid,'units','m/s');
netcdf.putAtt(ncid,eofid,'long name','LU velocity EOFs');
netcdf.putAtt(ncid,ucid,'units','m/s');
netcdf.putAtt(ncid,ucid,'long name','LU zonal mean correction');
netcdf.putAtt(ncid,vcid,'units','m/s');
netcdf.putAtt(ncid,vcid,'long name','LU meridional mean correction');
netcdf.putAtt(ncid,axxid,'units','m^2/s');
netcdf.putAtt(ncid,axxid,'long name','LU diffusion tensor (zonal)');
netcdf.putAtt(ncid,ayyid,'units','m^2/s');
netcdf.putAtt(ncid,ayyid,'long name','LU diffusion tensor (meridional)');
netcdf.putAtt(ncid,axyid,'units','m^2/s');
netcdf.putAtt(ncid,axyid,'long name','LU diffusion tensor (cross)');

% Leave define mode and enter data mode
netcdf.endDef(ncid);

% Write 1D data to variables
netcdf.putVar(ncid,mid,idm);
netcdf.putVar(ncid,xpid,xpo);
netcdf.putVar(ncid,ypid,ypo);
netcdf.putVar(ncid,zid,zo);
netcdf.putVar(ncid,xtid,xto);
netcdf.putVar(ncid,ytid,yto);

% Write 2D data to variables
start = [0 0]; count = [npo nmo];
netcdf.putVar(ncid,eofid,start,count,eofo);

% Write 3D data to variables
start = [0 0 0]; count = [nxpo nyto nlo];
netcdf.putVar(ncid,ucid,start,count,uco);
netcdf.putVar(ncid,axxid,start,count,axxo);

start = [0 0 0]; count = [nxto nypo nlo];
netcdf.putVar(ncid,vcid,start,count,vco);
netcdf.putVar(ncid,ayyid,start,count,ayyo);

start = [0 0 0]; count = [nxpo nypo nlo];
netcdf.putVar(ncid,axyid,start,count,axyo);

% Close the file
netcdf.close(ncid);

disp('Program terminates')