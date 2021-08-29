% Process POD of the data produced by '../coarse_grain'.
% The output file will be used in 'LU_proc.m'.
% 
% Written by Long Li 2020-08-10.
%

clear; close all; clc;

%% Set parameters
% base_dir is the home location of all Q-GCM data files.
% file_dir is the output location of all Q-GCM plots and other output.
% run_name is the name given for the folder of a particular run within
%             these locations.
% infile and outfile are the name of input and output NetCDF files.
%
% Raw Q-GCM data is stored in base_dir/run_name/.
% Processed data is placed in file_dir/run_name/.

% Input
base_dir = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/POD/data';
subs_dir = {'yrs105-106';'yrs106-107';'yrs107-108'; ...
            'yrs108-109';'yrs109-110';'yrs110-111'; ...
            'yrs111-112';'yrs112-113';'yrs113-114';'yrs114-115'};
infile = 'ocref80.nc'; 

% Output
file_dir = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/POD';
outfile = 'oceof80.nc';

%% Full data
% Go to the base directory and figure out which sub-directories
% the data is being held in.

files = fullfile(base_dir, subs_dir); 
nfiles = length(files);

if ~ (exist(file_dir,'dir')==7)
    mkdir(file_dir);
end

% Get dimensions from 1st file
file1 = fullfile(files{1},infile);
ncdisp(file1)
xpo = ncread(file1, 'xp'); xpo = xpo.*1e3; nxpo = length(xpo);
ypo = ncread(file1, 'yp'); ypo = ypo.*1e3; nypo = length(ypo);
xto = ncread(file1, 'xt'); xto = xto.*1e3; nxto = length(xto);
yto = ncread(file1, 'yt'); yto = yto.*1e3; nyto = length(yto);
zo  = ncread(file1, 'z');  zo  = zo.*1e3;  nlo  = length(zo);
dxo = xpo(2) - xpo(1);

%% Read data
% Collect the filtered snapshots and the time-mean of residuals 
% from the sub-directories. 
% A global time-mean will be returned at the end.

% Loop through each segment of run
tyrs = []; uo = []; vo = []; 
uro = []; vro = []; 
for i=1:nfiles   
    file1 = fullfile(files{i},infile);
    disp(['[ Opening ',file1,' ]'])
    n1 = min(i,2);
    % Time axis
    tmp = ncread(file1, 'time'); 
    tmp = tmp(n1:end); nt1 = length(tmp);
    tyrs = [tyrs; tmp]; clear tmp
    % Velocity fluctuations
    tmp = ncread(file1, 'ur', [1 1 1 n1], [Inf Inf Inf nt1]);
    uo = cat(4, uo, tmp); clear tmp
    tmp = ncread(file1, 'vr', [1 1 1 n1], [Inf Inf Inf nt1]);
    vo = cat(4, vo, tmp); clear tmp
end
disp(' ')
nto = length(tyrs);

%% POD procedure
% Perform the POD layer by layer. The eigenvalues, (orthonormal)
% spatial modes and (orthogonal) temporal modes will be returned.

disp('Processing POD');

lambda = zeros(nto,nlo); % eigenvalues (L2-norm of each mode)
umodes = 0.*uo; vmodes = umodes; % spatial modes
alpha = zeros(nto,nto,nlo); % temporal modes
umean = umodes(:,:,:,1); vmean = umean; % mean flows

for k=1:nlo
    fprintf(1,'Layer = %d\n',k);
    uk = squeeze(uo(:,:,k,:));
    vk = squeeze(vo(:,:,k,:));    
    [lambda(:,k), umodes(:,:,k,:), vmodes(:,:,k,:), ...
     alpha(:,:,k), umean(:,:,k), vmean(:,:,k)] = fct_POD(uk, vk); 
    clear uk vk
end
disp(' ')

% Noise mean 
uro = - umean;
vro = - vmean;
% The sign '-' emerges from the eddy-mean decomposition.

%% Save output file
% Save {'lambda','alpha','umodes','vmodes'} into 'outfile'.
% 'lambda' will be used to truncate the EOFs for the noise.
% 'alpha' is used to diagnose the temporal behavior of the EOFs.

disp('Save NetCDF data')

% Create new NetCDF file
tmp = fullfile(file_dir,outfile);
if exist(tmp,'file')==2
   delete(tmp);
end
cmode = netcdf.getConstant('NOCLOBBER');
cmode = bitor(cmode,netcdf.getConstant('64BIT_OFFSET'));
ncid = netcdf.create(tmp,cmode); clear tmp

% Define dimensions
tdim = netcdf.defDim(ncid,'hmod',nto);
xpdim = netcdf.defDim(ncid,'xp',nxpo);
ypdim = netcdf.defDim(ncid,'yp',nypo);
zdim = netcdf.defDim(ncid,'z',nlo);
xtdim = netcdf.defDim(ncid,'xt',nxto);
ytdim = netcdf.defDim(ncid,'yt',nyto);

% Define 1D variables
tid = netcdf.defVar(ncid,'hmod','NC_INT',tdim);
xpid = netcdf.defVar(ncid,'xp','NC_DOUBLE',xpdim);
ypid = netcdf.defVar(ncid,'yp','NC_DOUBLE',ypdim);
zid = netcdf.defVar(ncid,'z','NC_DOUBLE',zdim);
xtid = netcdf.defVar(ncid,'xt','NC_DOUBLE',xtdim);
ytid = netcdf.defVar(ncid,'yt','NC_DOUBLE',ytdim);

% Define 2D variables
dims = [tdim zdim];
lid = netcdf.defVar(ncid,'lambda','NC_DOUBLE',dims);

% Define 3D variables
dims = [tdim tdim zdim];
aid = netcdf.defVar(ncid,'alpha','NC_DOUBLE',dims);
dims = [xpdim ytdim zdim];
umid = netcdf.defVar(ncid,'umean','NC_DOUBLE',dims);
usid = netcdf.defVar(ncid,'usubs','NC_DOUBLE',dims);
dims = [xtdim ypdim zdim];
vmid = netcdf.defVar(ncid,'vmean','NC_DOUBLE',dims);
vsid = netcdf.defVar(ncid,'vsubs','NC_DOUBLE',dims);

% Define 4D variables
dims = [xpdim ytdim zdim tdim];
uid = netcdf.defVar(ncid,'umode','NC_DOUBLE',dims);
dims = [xtdim ypdim zdim tdim];
vid = netcdf.defVar(ncid,'vmode','NC_DOUBLE',dims);

% Define attributes
netcdf.putAtt(ncid,tid,'long name','Indices of horizontal modes');

netcdf.putAtt(ncid,xpid,'units','km');
netcdf.putAtt(ncid,xpid,'long name','X axis (p-grid)');
netcdf.putAtt(ncid,ypid,'units','km');
netcdf.putAtt(ncid,ypid,'long name','Y axis (p-grid)');

netcdf.putAtt(ncid,xtid,'units','km');
netcdf.putAtt(ncid,xtid,'long name','X axis (T-grid)');
netcdf.putAtt(ncid,ytid,'units','km');
netcdf.putAtt(ncid,ytid,'long name','Y axis (T-grid)');

netcdf.putAtt(ncid,zid,'units','km');
netcdf.putAtt(ncid,zid,'long name','Mid-layer depth axis');

netcdf.putAtt(ncid,lid,'units','m^2/s^2');
netcdf.putAtt(ncid,lid,'long name','Eigenvalues of covariance');

netcdf.putAtt(ncid,aid,'units','s');
netcdf.putAtt(ncid,aid,'long name','Temporal modes');

netcdf.putAtt(ncid,uid,'units','m/s');
netcdf.putAtt(ncid,uid,'long name','Zonal spatial modes');
netcdf.putAtt(ncid,vid,'units','m/s');
netcdf.putAtt(ncid,vid,'long name','Meridional spatial modes');

netcdf.putAtt(ncid,umid,'units','m/s');
netcdf.putAtt(ncid,umid,'long name','Zonal mean flow');
netcdf.putAtt(ncid,vmid,'units','m/s');
netcdf.putAtt(ncid,vmid,'long name','Meridional mean flow');

netcdf.putAtt(ncid,usid,'units','m/s');
netcdf.putAtt(ncid,usid,'long name','Zonal SGS fluctuation');
netcdf.putAtt(ncid,vsid,'units','m/s');
netcdf.putAtt(ncid,vsid,'long name','Meridional SGS fluctuation');

% Leave define mode and enter data mode
netcdf.endDef(ncid);

% Write 1D data to variables
netcdf.putVar(ncid,tid,1:nto);
netcdf.putVar(ncid,xpid,xpo.*1e-3);
netcdf.putVar(ncid,ypid,ypo.*1e-3);
netcdf.putVar(ncid,zid,zo.*1e-3);
netcdf.putVar(ncid,xtid,xto.*1e-3);
netcdf.putVar(ncid,ytid,yto.*1e-3);

% Write 2D data to variables
start = [0 0]; count = [nto nlo];
netcdf.putVar(ncid,lid,start,count,lambda(1:nto,:));

% Write 3D data to variables
start = [0 0 0]; count = [nto nto nlo];
netcdf.putVar(ncid,aid,start,count,alpha(1:nto,1:nto,:));
start = [0 0 0]; count = [nxpo nyto nlo];
netcdf.putVar(ncid,umid,start,count,umean);
netcdf.putVar(ncid,usid,start,count,uro);
start = [0 0 0]; count = [nxto nypo nlo];
netcdf.putVar(ncid,vmid,start,count,vmean);
netcdf.putVar(ncid,vsid,start,count,vro);

% Write 4D data to variables
start = [0 0 0 0]; count = [nxpo nyto nlo nto];
netcdf.putVar(ncid,uid,start,count,umodes(:,:,:,1:nto));
start = [0 0 0 0]; count = [nxto nypo nlo nto];
netcdf.putVar(ncid,vid,start,count,vmodes(:,:,:,1:nto));

% Close the file
netcdf.close(ncid);

disp('Program terminates')