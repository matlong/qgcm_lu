c***********************************************************************
c     Q-GCM Version 1.5.0 : last modified 18/09/2013
c***********************************************************************
c
c     Copyright 2013 Jeff Blundell, Andy Hogg and Bill Dewar.
c     This file is part of Q-GCM.
c
c     Q-GCM is free software: you can redistribute it and/or modify
c     it under the terms of the GNU General Public License as
c     published by the Free Software Foundation, either version 3
c     of the License, or (at your option) any later version.
c
c     Q-GCM is distributed in the hope that it will be useful,
c     but WITHOUT ANY WARRANTY; without even the implied warranty
c     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
c     See the GNU General Public License for more details.
c
c     You should have received a copy of the GNU General Public License
c     along with Q-GCM.  If not, see <http://www.gnu.org/licenses/>.
c
c***********************************************************************
c
      MODULE nc_subs

*     Contains subprograms for input, creation and output of files
*     in netCDF format, including ocean and atmosphere snapshot
*     dumps, reading and saving restart files, and error handling.
*     For files created by the subroutines in this program file,
*     all real fields are single precision (float) format, except for
*     restart dumps written by resave_nc and read in by restart_nc.
*
*     As all input/output from these routines is explicitly intended
*     to be in netCDF format, the corresponding #ifdef is used to
*     remove code to the widest possible extent, with configuration-
*     dependant #ifdefs being nested within the use_netcdf test.

      IMPLICIT NONE

      PRIVATE

      PUBLIC :: handle_err, ocnc_init, atnc_init, ocnc_out,
     &          atnc_out, resave_nc, restart_nc

#ifdef lu_mode
#  ifndef atmos_only
      PUBLIC :: read_ocpod, oclu_init, oclu_out
#  endif /* not atmos_only */      
#  ifndef ocean_only
      PUBLIC :: read_atpod, atlu_init, atlu_out
#  endif /* not ocean_only */      
#endif

#ifdef use_edvel
#  ifndef atmos_only
      PUBLIC :: read_ocedv
#  endif /* not atmos_only */      
#  ifndef ocean_only
      PUBLIC :: read_atedv
#  endif /* not ocean_only */      
#endif
      
      integer, PUBLIC, SAVE :: restid

#ifdef use_netcdf

#  ifndef ocean_only

*     Storage for identifiers for atmospheric netCDF dumps
      integer, PUBLIC, SAVE :: atpid, attid
      integer, PRIVATE, SAVE :: ast_id, pa_id, qa_id, eka_id, ha_id,
     &                  eta_id, txa_id, tya_id, timap_id, timat_id

#    ifdef lu_mode
      integer, PUBLIC, SAVE :: atpodid, atrid
      integer, PRIVATE, SAVE :: ura_id, vra_id, aaxx_id, aayy_id, 
     &                          aaxy_id, tima_id 
#    endif /* lu_mode */

#    ifdef use_edvel
      integer, PUBLIC, SAVE :: atedvid
#    endif /* use_edvel */

#  endif /* not ocean_only */

#  ifndef atmos_only

*     Storage for identifiers for oceanic netCDF dumps
      integer, PUBLIC, SAVE :: ocpid, octid
      integer, PRIVATE, SAVE :: sst_id, po_id, qo_id, eko_id, ho_id,
     &                          txo_id, tyo_id, timop_id, timot_id

#    ifdef lu_mode
      integer, PUBLIC, SAVE :: ocpodid, ocrid
      integer, PRIVATE, SAVE :: uro_id, vro_id, aoxx_id, aoyy_id, 
     &                          aoxy_id, timo_id 
#    endif /* lu_mode */

#    ifdef use_edvel
      integer, PUBLIC, SAVE :: ocedvid
#    endif /* use_edvel */

#  endif /* not atmos_only */

#endif /* use_netcdf */

      CONTAINS

c***********************************************************************
c
      SUBROUTINE handle_err (ncstat, fromst)

      IMPLICIT NONE
#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments
      integer, INTENT(IN) :: ncstat
      character (len=*), INTENT(IN), OPTIONAL :: fromst
!!    fromst is an optional string indicating where the call came
!!    from that caused the netCDF problem (e.g. subroutine name)

#ifdef use_netcdf
*     Local variables

!!  Routine which interprets errors from netCDF output functions,
!!  prints them to standard output and then kills the whole run.
      if ( ncstat.ne.NF_NOERR ) then
        if ( present(fromst) ) then
          print *, trim(fromst)//':'//trim( nf_strerror(ncstat) )
         else
          print *, trim( nf_strerror(ncstat) )
        endif
        stop 'netCDF:: STOPPED'
      endif
#endif /* use_netcdf */

      END SUBROUTINE handle_err

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE ocnc_init (nsko, outfloc)

*     Initialises the files ocpo.nc (p-grid, id = ocpid)
*     and ocsst.nc (T-grid, id = octid)

*     Modules
      USE parameters
#ifndef atmos_only
      USE occonst
#endif /* not atmos_only */
      USE timinfo, ONLY : noutstepoc

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments
      integer nsko, outfloc(7)

#ifdef use_netcdf
#  ifndef atmos_only

*     Local parameters
      character (len=*), parameter :: subnam = 'ocnc_init'

!!  netCDF variables used locally
      integer ncstat, timopdim, xopdim, yopdim, lodim, lomdim
      integer timotdim, xotdim, yotdim, axodims, odims(3), podims(4)
      integer xop_id, yop_id, xot_id, yot_id, lo_id, lom_id

!!  Other variables used locally
      double precision xo(nxpo),yo(nypo),tmp(nlo)
      integer i, iwk, mwk

!!    Definition section: define dimensions and variables
!!    ===================================================
!!    Define four dimensions: x, y, z, time

!!    Dimension definitions for p-grid output file
!!    --------------------------------------------
      ncstat = nf_def_dim(ocpid, 'time', noutstepoc, timopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    x and y dimensions for the (subsampled) p-grid
      mwk = mod(nxpo, nsko)
      iwk = min(mwk,1) + (nxpo-mwk)/nsko
      ncstat = nf_def_dim(ocpid, 'xp', iwk, xopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      mwk = mod(nypo, nsko)
      iwk = min(mwk,1) + (nypo-mwk)/nsko
      ncstat = nf_def_dim(ocpid, 'yp', iwk, yopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  And here the z dimension
      ncstat = nf_def_dim(ocpid, 'z', nlo, lodim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Also define a zi dimension for the interfaces
      ncstat = nf_def_dim(ocpid, 'zi', nlo-1, lomdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Dimension definitions for T-grid output file
!!    --------------------------------------------
      ncstat = nf_def_dim(octid, 'time', noutstepoc, timotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    x and y dimensions for the (subsampled) T-grid
      mwk = mod(nxto, nsko)
      iwk = min(mwk,1) + (nxto-mwk)/nsko
      ncstat = nf_def_dim(octid, 'xt', iwk, xotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      mwk = mod(nyto, nsko)
      iwk = min(mwk,1) + (nyto-mwk)/nsko
      ncstat = nf_def_dim(octid, 'yt', iwk, yotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for p-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xp' which stores x gridpoints (p-grid)
      axodims = xopdim
!!      ncstat = nf_def_var(ocpid, 'xp', NF_DOUBLE, 1, axodims, xop_id)
      ncstat = nf_def_var(ocpid, 'xp', NF_FLOAT, 1, axodims, xop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, xop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, xop_id, 'long_name',
     &            21, 'Ocean X axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yp' which stores y gridpoints (p-grid)
      axodims = yopdim
!!      ncstat = nf_def_var(ocpid, 'yp', NF_DOUBLE, 1, axodims, yop_id)
      ncstat = nf_def_var(ocpid, 'yp', NF_FLOAT, 1, axodims, yop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, yop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, yop_id, 'long_name',
     &            21, 'Ocean Y axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time data in years
      axodims = timopdim
!!      ncstat=nf_def_var(ocpid, 'time', NF_DOUBLE, 1, axodims, timop_id)
      ncstat=nf_def_var(ocpid, 'time', NF_FLOAT, 1, axodims, timop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, timop_id, 'units',5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(ocpid,timop_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'z'
!!  to store layer thicknesses; p-grid only
      axodims = lodim
!!      ncstat = nf_def_var(ocpid, 'z', NF_DOUBLE, 1, axodims, lo_id)
      ncstat = nf_def_var(ocpid, 'z', NF_FLOAT, 1, axodims, lo_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, lo_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, lo_id, 'long_name',
     &            26, 'Ocean mid-layer depth axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'zi'
!!  to store interface heights; p-grid only
      axodims = lomdim
!!      ncstat = nf_def_var(ocpid, 'zi', NF_DOUBLE, 1, axodims, lom_id)
      ncstat = nf_def_var(ocpid, 'zi', NF_FLOAT, 1, axodims, lom_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, lom_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocpid, lom_id, 'long_name',
     &            26, 'Ocean interface depth axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for T-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xt' which stores x gridpoints (T-grid)
      axodims = xotdim
!!      ncstat = nf_def_var(octid, 'xt', NF_DOUBLE, 1, axodims, xot_id)
      ncstat = nf_def_var(octid, 'xt', NF_FLOAT, 1, axodims, xot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(octid, xot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(octid, xot_id, 'long_name',
     &            21, 'Ocean X axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yt' which stores y gridpoints (T-grid)
      axodims = yotdim
!!      ncstat = nf_def_var(octid, 'yt', NF_DOUBLE, 1, axodims, yot_id)
      ncstat = nf_def_var(octid, 'yt', NF_FLOAT, 1, axodims, yot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(octid, yot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(octid, yot_id, 'long_name',
     &            21, 'Ocean Y axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time data in years
      axodims = timotdim
!!      ncstat=nf_def_var(octid, 'time', NF_DOUBLE, 1, axodims, timot_id)
      ncstat=nf_def_var(octid, 'time', NF_FLOAT, 1, axodims, timot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(octid, timot_id, 'units', 5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(octid,timot_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'sst'
!!  to store ocean surface temperatures
      if ( outfloc(1).eq.1 ) then
        odims(1) = xotdim
        odims(2) = yotdim
        odims(3) = timotdim
!!        ncstat = nf_def_var(octid, 'sst', NF_DOUBLE, 3, odims, sst_id)
        ncstat = nf_def_var(octid, 'sst', NF_FLOAT, 3, odims, sst_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining sst in ocean T-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(octid, sst_id, 'units', 1, 'K')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(octid, sst_id, 'long_name',
     &              25, 'Ocean surface temperature')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'p' to store pressure
      if ( outfloc(2).eq.1 ) then
        podims(1) = xopdim
        podims(2) = yopdim
        podims(3) = lodim
        podims(4) = timopdim
!!        ncstat = nf_def_var(ocpid, 'p', NF_DOUBLE, 4, podims, po_id)
        ncstat = nf_def_var(ocpid, 'p', NF_FLOAT, 4, podims, po_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining p in ocean p-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(ocpid, po_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(ocpid, po_id, 'long_name',
     &              22, 'Ocean dynamic pressure')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'q' to store potential vorticity
      if ( outfloc(3).eq.1 ) then
        podims(1) = xopdim
        podims(2) = yopdim
        podims(3) = lodim
        podims(4) = timopdim
!!        ncstat = nf_def_var(ocpid, 'q', NF_DOUBLE, 4, podims, qo_id)
        ncstat = nf_def_var(ocpid, 'q', NF_FLOAT, 4, podims, qo_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining q in ocean p-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(ocpid, qo_id, 'units', 4, 's^-1')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(ocpid, qo_id, 'long_name',
     &                 25, 'Ocean potential vorticity')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 3-dimensional variable called 'wekt'
!!  to store oceanic Ekman velocity (on T-grid)
      if ( outfloc(4).eq.1 ) then
        odims(1) = xotdim
        odims(2) = yotdim
        odims(3) = timotdim
!!        ncstat = nf_def_var(octid, 'wekt', NF_DOUBLE, 3, odims, eko_id)
        ncstat = nf_def_var(octid, 'wekt', NF_FLOAT, 3, odims, eko_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining wekt in ocean T-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(octid, eko_id, 'units', 3, 'm/s')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(octid, eko_id, 'long_name',
     &              29, 'Ocean Ekman velocity (T-grid)')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'h'
!!  to store oceanic interface displacements
      if ( outfloc(5).eq.1 ) then
        podims(1) = xopdim
        podims(2) = yopdim
        podims(3) = lomdim
        podims(4) = timopdim
!!        ncstat = nf_def_var(ocpid, 'h', NF_DOUBLE, 4, podims, ho_id)
        ncstat = nf_def_var(ocpid, 'h', NF_FLOAT, 4, podims, ho_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining h in ocean p-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(ocpid, ho_id, 'units', 1, 'm')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(ocpid, ho_id, 'long_name',
     &              28, 'Ocean interface displacement')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define two 3-dimensional variables called 'taux' and
!!  'tauy' to store ocean components of dynamic windstress
      if ( outfloc(6).eq.1 ) then
        odims(1) = xopdim
        odims(2) = yopdim
        odims(3) = timopdim
!!        ncstat = nf_def_var(ocpid, 'taux', NF_DOUBLE, 3, odims, txo_id)
        ncstat = nf_def_var(ocpid, 'taux', NF_FLOAT, 3, odims, txo_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining taux in ocean p-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(ocpid, txo_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(ocpid, txo_id, 'long_name',
     &              25, 'Zonal dynamic wind stress')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        
!!        ncstat = nf_def_var(ocpid, 'tauy', NF_DOUBLE, 3, odims, tyo_id)
        ncstat = nf_def_var(ocpid, 'tauy', NF_FLOAT, 3, odims, tyo_id)
        if ( ncstat.ne.NF_NOERR ) then
          print *,' problem defining tauy in ocean p-grid file'
          call handle_err (ncstat, subnam)
        endif
        ncstat = nf_put_att_text(ocpid, tyo_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(ocpid, tyo_id, 'long_name',
     &              30, 'Meridional dynamic wind stress')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  The oceanic mixed layer thickness is fixed in this
!!  version of the model, so outfloc(7) is irrelevant

!!  Leave definition mode: entering data mode
      ncstat = nf_enddef(ocpid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' ocnc_init: enddef problem for ocean p-grid file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_enddef(octid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' ocnc_init: enddef problem for ocean T-grid file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif

!!  Calculate x gridpoints and store in 'x' arrays
!!  p-grid points
      mwk = mod(nxpo, nsko)
      iwk = min(mwk,1) + (nxpo-mwk)/nsko
      do i=1,iwk
        xo(i) = 1.0d-3*( xpo(1+(i-1)*nsko) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(ocpid, xop_id, 1, iwk, xo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      mwk = mod(nxto, nsko)
      iwk = min(mwk,1) + (nxto-mwk)/nsko
      do i=1,iwk
        xo(i) = 1.0d-3*( xto(1+(i-1)*nsko) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(octid, xot_id, 1, iwk, xo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Calculate y gridpoints and store in 'y' arrays
!!  p-grid points
      mwk = mod(nypo, nsko)
      iwk = min(mwk,1) + (nypo-mwk)/nsko
      do i=1,iwk
        yo(i) = 1.0d-3*( ypo(1+(i-1)*nsko) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(ocpid, yop_id, 1, iwk, yo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      mwk = mod(nyto, nsko)
      iwk = min(mwk,1) + (nyto-mwk)/nsko
      do i=1,iwk
        yo(i) = 1.0d-3*( yto(1+(i-1)*nsko) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(octid, yot_id, 1, iwk, yo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert mid-layer depths into km and store in 'z'
      tmp(1) = 0.5d-3*hoc(1)
      do i=2,nlo
        tmp(i) = tmp(i-1) + 0.5d-3*( hoc(i-1) + hoc(i) )
      enddo
      ncstat = nf_put_var_double(ocpid, lo_id, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!!!! put operation seems to lack some arguments!!!
!!!!! This may be because it's a put_var not put_vara!!!

!!  Convert interface depths into km and store in 'zi'
      tmp(1) = 1.0d-3*hoc(1)
      do i=2,nlo-1
        tmp(i) = tmp(i-1) + 1.0d-3*hoc(i)
      enddo
      ncstat = nf_put_vara_double(ocpid, lom_id, 1, nlo-1, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#  endif /* not atmos_only */
#endif /* use_netcdf */

      END SUBROUTINE ocnc_init

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE atnc_init (nska, outflat)

*     Modules
      USE parameters
#ifndef ocean_only
      USE atconst
#endif
      USE timinfo, ONLY : noutstepat

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments
      integer nska, outflat(7)

#ifdef use_netcdf
#  ifndef ocean_only

*     Local parameters
      character (len=*), parameter :: subnam = 'atnc_init'

!!  netCDF variables used locally
      integer ncstat, timapdim, xapdim, yapdim, ladim, lamdim
      integer timatdim, xatdim, yatdim, axadims, adims(3), padims(4)
      integer xap_id, yap_id, xat_id, yat_id, la_id, lam_id

!!  Other variables used locally
      double precision xa(nxpa),ya(nypa),tmp(nla)
      integer i, iwk, mwk

!!  Define four dimensions: x, y, z, time
!!  Do pressure fields first
      ncstat = nf_def_dim(atpid, 'time', noutstepat, timapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Note that x and y are designed for subsampling
!!  Here are the x and y dimensions for the p-grid
      mwk = mod(nxpa, nska)
      iwk = min(mwk,1) + (nxpa-mwk)/nska
      ncstat = nf_def_dim(atpid, 'xp', iwk, xapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      mwk = mod(nypa, nska)
      iwk = min(mwk,1) + (nypa-mwk)/nska
      ncstat = nf_def_dim(atpid, 'yp', iwk, yapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  And here the z dimension
      ncstat = nf_def_dim(atpid, 'z', nla, ladim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Also define a zi dimension for the interfaces
      ncstat = nf_def_dim(atpid, 'zi', nla-1, lamdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Now do it for the T-grid
      ncstat = nf_def_dim(attid, 'time', noutstepat, timatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Here are the x and y dimensions for the T-grid
      mwk = mod(nxta, nska)
      iwk = min(mwk,1) + (nxta-mwk)/nska
      ncstat = nf_def_dim(attid, 'xt', iwk, xatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      mwk = mod(nyta, nska)
      iwk = min(mwk,1) + (nyta-mwk)/nska
      ncstat = nf_def_dim(attid, 'yt', iwk, yatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xp' which stores x gridpoints (p-grid)
      axadims = xapdim
!!      ncstat = nf_def_var(atpid, 'xp', NF_DOUBLE, 1, axadims, xap_id)
      ncstat = nf_def_var(atpid, 'xp', NF_FLOAT, 1, axadims, xap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, xap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, xap_id, 'long_name',
     &            26, 'Atmosphere X axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xt' which stores x gridpoints (T-grid)
      axadims = xatdim
!!      ncstat = nf_def_var(attid, 'xt', NF_DOUBLE, 1, axadims, xat_id)
      ncstat = nf_def_var(attid, 'xt', NF_FLOAT, 1, axadims, xat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(attid, xat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(attid, xat_id, 'long_name',
     &            26, 'Atmosphere X axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yp' which stores y gridpoints (p-grid)
      axadims = yapdim
!!      ncstat = nf_def_var(atpid, 'yp', NF_DOUBLE, 1, axadims, yap_id)
      ncstat = nf_def_var(atpid, 'yp', NF_FLOAT, 1, axadims, yap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, yap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, yap_id, 'long_name',
     &            26, 'Atmosphere Y axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yt' which stores y gridpoints (T-grid)
      axadims = yatdim
!!      ncstat = nf_def_var(attid, 'yt', NF_DOUBLE, 1, axadims, yat_id)
      ncstat = nf_def_var(attid, 'yt', NF_FLOAT, 1, axadims, yat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(attid, yat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(attid, yat_id, 'long_name',
     &            26, 'Atmosphere Y axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time data in years
!!  p-grid:
      axadims = timapdim
!!      ncstat=nf_def_var(atpid, 'time', NF_DOUBLE, 1, axadims, timap_id)
      ncstat=nf_def_var(atpid, 'time', NF_FLOAT, 1, axadims, timap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, timap_id, 'units', 5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(atpid,timap_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid:
      axadims = timatdim
!!      ncstat=nf_def_var(attid, 'time', NF_DOUBLE, 1, axadims, timat_id)
      ncstat=nf_def_var(attid, 'time', NF_FLOAT, 1, axadims, timat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(attid, timat_id, 'units', 5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(attid,timat_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'z'
!!  to store layer thicknesses; p-grid only
      axadims = ladim
!!      ncstat = nf_def_var(atpid, 'z', NF_DOUBLE, 1, axadims, la_id)
      ncstat = nf_def_var(atpid, 'z', NF_FLOAT, 1, axadims, la_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, la_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, la_id, 'long_name',
     &            32, 'Atmosphere mid-layer height axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'zi'
!!  to store interface heights; p-grid only
      axadims = lamdim
!!      ncstat = nf_def_var(atpid, 'zi', NF_DOUBLE, 1, axadims, lam_id)
      ncstat = nf_def_var(atpid, 'zi', NF_FLOAT, 1, axadims, lam_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, lam_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atpid, lam_id, 'long_name',
     &            32, 'Atmosphere interface height axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'ast'
!!  to store atmosphere surface temperatures
      if ( outflat(1).eq.1 ) then
        adims(1) = xatdim
        adims(2) = yatdim
        adims(3) = timatdim
!!        ncstat = nf_def_var(attid, 'ast', NF_DOUBLE, 3, adims, ast_id)
        ncstat = nf_def_var(attid, 'ast', NF_FLOAT, 3, adims, ast_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, ast_id, 'units', 1, 'K')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, ast_id, 'long_name',
     &              30, 'Atmosphere surface temperature')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'p' to store pressure
      if ( outflat(2).eq.1 ) then
        padims(1) = xapdim
        padims(2) = yapdim
        padims(3) = ladim
        padims(4) = timapdim
!!        ncstat = nf_def_var(atpid, 'p', NF_DOUBLE, 4, padims, pa_id)
        ncstat = nf_def_var(atpid, 'p', NF_FLOAT, 4, padims, pa_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, pa_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, pa_id, 'long_name',
     &              27, 'Atmosphere dynamic pressure')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'q' to store potential vorticity
      if ( outflat(3).eq.1 ) then
        padims(1) = xapdim
        padims(2) = yapdim
        padims(3) = ladim
        padims(4) = timapdim
!!        ncstat = nf_def_var(atpid, 'q', NF_DOUBLE, 4, padims, qa_id)
        ncstat = nf_def_var(atpid, 'q', NF_FLOAT, 4, padims, qa_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, qa_id, 'units', 4, 's^-1')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, qa_id, 'long_name',
     &              30, 'Atmosphere potential vorticity')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 3-dimensional variable called 'wekt'
!!  to store atmospheric Ekman velocity (on T-grid)
      if ( outflat(4).eq.1 ) then
        adims(1) = xatdim
        adims(2) = yatdim
        adims(3) = timatdim
!!        ncstat = nf_def_var(attid, 'wekt', NF_DOUBLE, 3, adims, eka_id)
        ncstat = nf_def_var(attid, 'wekt', NF_FLOAT, 3, adims, eka_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, eka_id, 'units', 3, 'm/s')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, eka_id, 'long_name',
     &              34, 'Atmosphere Ekman velocity (T-grid)')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 4-dimensional variable called 'h'
!!  to store atmospheric interface displacement
      if ( outflat(5).eq.1 ) then
         padims(1) = xapdim
         padims(2) = yapdim
         padims(3) = lamdim
         padims(4) = timapdim
!!         ncstat = nf_def_var(atpid, 'h', NF_DOUBLE, 4, padims, ha_id)
         ncstat = nf_def_var(atpid, 'h', NF_FLOAT, 4, padims, ha_id)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
         ncstat = nf_put_att_text(atpid, ha_id, 'units', 1, 'm')
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
         ncstat = nf_put_att_text(atpid, ha_id, 'long_name',
     &               33, 'Atmosphere interface displacement')
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define two 3-dimensional variables called 'taux' and 'tauy'
!!  to store atmosphere components of dynamic windstress
      if ( outflat(6).eq.1 ) then
        adims(1) = xapdim
        adims(2) = yapdim
        adims(3) = timapdim
!!        ncstat = nf_def_var(atpid, 'taux', NF_DOUBLE, 3, adims, txa_id)
        ncstat = nf_def_var(atpid, 'taux', NF_FLOAT, 3, adims, txa_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, txa_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, txa_id, 'long_name',
     &              17, 'Zonal wind stress')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!        ncstat = nf_def_var(atpid, 'tauy', NF_DOUBLE, 3, adims, tya_id)
        ncstat = nf_def_var(atpid, 'tauy', NF_FLOAT, 3, adims, tya_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, tya_id, 'units', 7, 'm^2/s^2')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(atpid, tya_id, 'long_name',
     &              22, 'Meridional wind stress')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Define a 3-dimensional variable called 'hmixa'
!!  to store atmosphere mixed layer height
      if ( outflat(7).eq.1 ) then
        adims(1) = xatdim
        adims(2) = yatdim
        adims(3) = timatdim
!!        ncstat=nf_def_var(attid, 'hmixa', NF_DOUBLE, 3, adims, eta_id)
        ncstat=nf_def_var(attid, 'hmixa', NF_FLOAT, 3, adims, eta_id)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, eta_id, 'units', 1, 'm')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
        ncstat = nf_put_att_text(attid, eta_id, 'long_name',
     &              28, 'Mixed layer interface height')
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Leave definition mode: entering data mode
      ncstat = nf_enddef(atpid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' atnc_init: enddef problem for atmos p-grid file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_enddef(attid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' atnc_init: enddef problem for atmos T-grid file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif

!!  Calculate x gridpoints and store in 'x' arrays
!!  p-grid points
      mwk = mod(nxpa, nska)
      iwk = min(mwk,1) + (nxpa-mwk)/nska
      do i=1,iwk
        xa(i) = 1.0d-3*xpa(1+(i-1)*nska)
      enddo
      ncstat = nf_put_vara_double(atpid, xap_id, 1, iwk, xa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      mwk = mod(nxta, nska)
      iwk = min(mwk,1) + (nxta-mwk)/nska
      do i=1,iwk
        xa(i) = 1.0d-3*xta(1+(i-1)*nska)
      enddo
      ncstat = nf_put_vara_double(attid, xat_id, 1, iwk, xa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Calculate y gridpoints and store in 'y' arrays
!!  p-grid points
      mwk = mod(nypa, nska)
      iwk = min(mwk,1) + (nypa-mwk)/nska
      do i=1,iwk
        ya(i) = 1.0d-3*ypa(1+(i-1)*nska)
      enddo
      ncstat = nf_put_vara_double(atpid, yap_id, 1, iwk, ya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      mwk = mod(nyta, nska)
      iwk = min(mwk,1) + (nyta-mwk)/nska
      do i=1,iwk
        ya(i) = 1.0d-3*yta(1+(i-1)*nska)
      enddo
      ncstat = nf_put_vara_double(attid, yat_id, 1, iwk, ya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert mid-layer heights into km and store in 'z'
      tmp(1) = 0.5d-3*hat(1)
      do i=2,nla
        tmp(i) = tmp(i-1) + 0.5d-3*( hat(i-1) + hat(i) )
      enddo
      ncstat = nf_put_var_double(atpid,la_id, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert interface heights into km and store in 'zi'
      tmp(1) = 1.0d-3*hat(1)
      do i=2,nla-1
        tmp(i) = tmp(i-1) + 1.0d-3*hat(i)
      enddo
      ncstat = nf_put_var_double(atpid, lam_id, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#  endif /* not ocean_only */
#endif /* use_netcdf */

      END SUBROUTINE atnc_init

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE ocnc_out (nsko, outfloc)

*     Puts data into the files ocpo.nc (p-grid, id = ocpid)
*     and ocsst.nc (T-grid, id = octid)

*     Modules
      USE parameters
#ifdef use_netcdf
#  ifndef atmos_only
      USE occonst
      USE ocstate
      USE intrfac, ONLY : sst, tauxo, tauyo
#  endif /* not atmos_only */
#endif /* use_netcdf */
      USE timinfo, ONLY : ntdone, noutoc, tyrs

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments
      integer nsko, outfloc(7)

#ifdef use_netcdf
#  ifndef atmos_only

*     Local parameters
      character (len=*), parameter :: subnam = 'ocnc_out'

!!  netCDF variables used locally
!!    start3, count3 are for 3D variables; start4, count4 for 4D
      integer ncstat,startt,countt,start3(3),count3(3),
     &        start4(4),count4(4)

!!  Other variables used locally
      double precision wrk(nxpo*nypo*nlo),rgpoc
      integer i,j,k,ipwk,jpwk,itwk,jtwk,mwk

!!  Store current time as part of 'time' vector
      startt = ntdone/noutoc + 1
      countt = 1
!!  Pressure file:
      ncstat=nf_put_vara_double(ocpid, timop_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Temperature file:
      ncstat=nf_put_vara_double(octid, timot_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Compute subsampling array indices
      mwk = mod(nxpo, nsko)
      ipwk = min(mwk,1) + (nxpo-mwk)/nsko
      mwk = mod(nxto, nsko)
      itwk = min(mwk,1) + (nxto-mwk)/nsko
      mwk = mod(nypo, nsko)
      jpwk = min(mwk,1) + (nypo-mwk)/nsko
      mwk = mod(nyto, nsko)
      jtwk = min(mwk,1) + (nyto-mwk)/nsko

!!  Store SST in 'sst'
      if (outfloc(1).eq.1) then
         start3(1) = 1
         start3(2) = 1
         start3(3) = ntdone/noutoc + 1
         count3(1) = itwk
         count3(2) = jtwk
         count3(3) = 1

!!  Looks ridiculous, but we now write out the sub-sampled
!!  sst array into wrk which is a very long vector.
!!  One needs to be careful about the order things are written
!!  out to get meaningful netCDF data here....
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (itwk,jtwk,wrk,sst,nsko)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jtwk
            do i=1,itwk
               wrk(i+itwk*(j-1)) = sst(1+(i-1)*nsko,1+(j-1)*nsko)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat=nf_put_vara_double(octid, sst_id, start3, count3, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Interpolate pressure onto T-grid points by averaging,
!!  and then write to 'p'. Note that we once again use the
!!  long wrk vector here.
      if ( outfloc(2).eq.1 ) then
         start4(1) = 1
         start4(2) = 1
         start4(3) = 1
         start4(4) = ntdone/noutoc + 1
         count4(1) = ipwk
         count4(2) = jpwk
         count4(3) = nlo
         count4(4) = 1
         do k=1,nlo
!$OMP       PARALLEL DO DEFAULT (NONE)
!$OMP&               PRIVATE (i,j)
!$OMP&               SHARED  (ipwk,jpwk,wrk,po,nsko,k)
!$OMP&               SCHEDULE (STATIC)
            do j=1,jpwk
               do i=1,ipwk
                  wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1))=
     &                 po(1+(i-1)*nsko,1+(j-1)*nsko,k)
               enddo
            enddo
!$OMP       END PARALLEL DO
         enddo
         ncstat = nf_put_vara_double(ocpid, po_id, start4, count4, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Subsample vorticity and then write to 'q'. Note
!!  that we once again use the long wrk vector here.
      if ( outfloc(3).eq.1 ) then
         start4(1) = 1
         start4(2) = 1
         start4(3) = 1
         start4(4) = ntdone/noutoc + 1
         count4(1) = ipwk
         count4(2) = jpwk
         count4(3) = nlo
         count4(4) = 1
         do k=1,nlo
!$OMP       PARALLEL DO DEFAULT (NONE)
!$OMP&               PRIVATE (i,j)
!$OMP&               SHARED  (ipwk,jpwk,wrk,qo,nsko,k)
!$OMP&               SCHEDULE (STATIC)
            do j=1,jpwk
               do i=1,ipwk
                  wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1))=
     &                 qo(1+(i-1)*nsko,1+(j-1)*nsko,k)
               enddo
            enddo
!$OMP       END PARALLEL DO
         enddo
         ncstat = nf_put_vara_double(ocpid, qo_id, start4, count4, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store wekto (Ekman velocity on T-grid) in 'wekt'
      if (outfloc(4).eq.1) then
         start3(1) = 1
         start3(2) = 1
         start3(3) = ntdone/noutoc + 1
         count3(1) = itwk
         count3(2) = jtwk
         count3(3) = 1
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (itwk,jtwk,wrk,wekto,nsko)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jtwk
            do i=1,itwk
               wrk(i+itwk*(j-1)) = wekto(1+(i-1)*nsko,1+(j-1)*nsko)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat=nf_put_vara_double(octid, eko_id, start3, count3, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store interface height in 'h'
      if (outfloc(5).eq.1) then
        start4(1) = 1
        start4(2) = 1
        start4(3) = 1
        start4(4) = ntdone/noutoc + 1
        count4(1) = ipwk
        count4(2) = jpwk
        count4(3) = nlo-1
        count4(4) = 1
        do k=1,nlo-1
          rgpoc = 1.0d0/gpoc(k)
!$OMP     PARALLEL DO DEFAULT (NONE)
!$OMP&             PRIVATE (i,j)
!$OMP&             SHARED  (ipwk,jpwk,wrk,rgpoc,po,nsko,k)
!$OMP&             SCHEDULE (STATIC)
          do j=1,jpwk
            do i=1,ipwk
              wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1)) =
     &                   rgpoc*(  po(1+(i-1)*nsko,1+(j-1)*nsko,k+1)
     &                          - po(1+(i-1)*nsko,1+(j-1)*nsko, k ) )
            enddo
          enddo
!$OMP     END PARALLEL DO
        enddo
        ncstat = nf_put_vara_double(ocpid, ho_id, start4, count4, wrk)
        if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store tauxo, tauyo (dynamic windstress components) in 'taux' and 'tauy'
      if (outfloc(6).eq.1) then
         start3(1) = 1
         start3(2) = 1
         start3(3) = ntdone/noutoc + 1
         count3(1) = ipwk
         count3(2) = jpwk
         count3(3) = 1
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (ipwk,jpwk,wrk,tauxo,nsko)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jpwk
            do i=1,ipwk
               wrk(i+ipwk*(j-1)) = tauxo(1+(i-1)*nsko,1+(j-1)*nsko)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat=nf_put_vara_double(ocpid, txo_id, start3, count3, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (ipwk,jpwk,wrk,tauyo,nsko)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jpwk
            do i=1,ipwk
               wrk(i+ipwk*(j-1)) = tauyo(1+(i-1)*nsko,1+(j-1)*nsko)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat=nf_put_vara_double(ocpid, tyo_id, start3, count3, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  The oceanic mixed layer thickness is fixed in this
!!  version of the model, so outfloc(7) is irrelevant

#  endif /* not atmos_only */
#endif /* use_netcdf */

      END SUBROUTINE ocnc_out

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE atnc_out (nska, outflat)

*     Modules
      USE parameters
#ifdef use_netcdf
#  ifndef ocean_only
      USE atconst
      USE atstate
      USE intrfac, ONLY : ast, tauxa, tauya, hmixa
#  endif /* not ocean_only */
#endif /* use_netcdf */
      USE timinfo, ONLY : ntdone, noutat, tyrs

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments
      integer nska, outflat(7)

#ifdef use_netcdf
#  ifndef ocean_only

*     Local parameters
      character (len=*), parameter :: subnam = 'atnc_out'

!!  netCDF variables used locally
      integer ncstat, startt, countt, start(3), count(3),
     &     startp(4), countp(4)

!!  Other variables used locally
      double precision wrk(nxpa*nypa*nla)
      integer i,j,k,ipwk,jpwk,itwk,jtwk,mwk

!!  Store current time as part of 'time' vector
      startt = ntdone/noutat + 1
      countt = 1
!!  Pressure file:
      ncstat=nf_put_vara_double(atpid, timap_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Temperature file:
      ncstat=nf_put_vara_double(attid, timat_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Compute subsampling array indices
      mwk = mod(nxpa, nska)
      ipwk = min(mwk,1) + (nxpa-mwk)/nska
      mwk = mod(nxta, nska)
      itwk = min(mwk,1) + (nxta-mwk)/nska
      mwk = mod(nypa, nska)
      jpwk = min(mwk,1) + (nypa-mwk)/nska
      mwk = mod(nyta, nska)
      jtwk = min(mwk,1) + (nyta-mwk)/nska

!!  Store AST in 'ast'
      if (outflat(1).eq.1) then
         start(1) = 1
         start(2) = 1
         start(3) = ntdone/noutat + 1
         count(1) = itwk
         count(2) = jtwk
         count(3) = 1

!!  Looks ridiculous, but we now write out the sub-sampled
!!  ast array into wrk which is a very long vector.
!!  One needs to be careful about the order things are written
!!  out to get meaningful netCDF data here....
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (itwk,jtwk,wrk,ast,nska)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jtwk
            do i=1,itwk
               wrk(i+itwk*(j-1)) = ast(1+(i-1)*nska,1+(j-1)*nska)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat = nf_put_vara_double(attid, ast_id, start, count, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Interpolate pressure onto T-grid points by averaging,
!!  and then write to 'p'. Note that we once again use the
!!  long wrk vector here.
      if (outflat(2).eq.1) then
         startp(1) = 1
         startp(2) = 1
         startp(3) = 1
         startp(4) = ntdone/noutat + 1
         countp(1) = ipwk
         countp(2) = jpwk
         countp(3) = nla
         countp(4) = 1
         do k=1,nla
!$OMP       PARALLEL DO DEFAULT (NONE)
!$OMP&               PRIVATE (i,j)
!$OMP&               SHARED  (ipwk,jpwk,wrk,pa,nska,k)
!$OMP&               SCHEDULE (STATIC)
            do j=1,jpwk
               do i=1,ipwk
                  wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1))=
     &                 pa(1+(i-1)*nska,1+(j-1)*nska,k)
               enddo
            enddo
!$OMP       END PARALLEL DO
         enddo
         ncstat=nf_put_vara_double(atpid, pa_id, startp, countp, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Interpolate vorticity onto T-grid points by averaging,
!!  and then write to 'q'. Note that we once again use the
!!  long wrk vector here.
      if (outflat(3).eq.1) then
         startp(1) = 1
         startp(2) = 1
         startp(3) = 1
         startp(4) = ntdone/noutat + 1
         countp(1) = ipwk
         countp(2) = jpwk
         countp(3) = nla
         countp(4) = 1
         do k=1,nla
!$OMP       PARALLEL DO DEFAULT (NONE)
!$OMP&               PRIVATE (i,j)
!$OMP&               SHARED  (ipwk,jpwk,wrk,qa,nska,k)
!$OMP&               SCHEDULE (STATIC)
            do j=1,jpwk
               do i=1,ipwk
                  wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1))=
     &                 qa(1+(i-1)*nska,1+(j-1)*nska,k)
               enddo
            enddo
!$OMP       END PARALLEL DO
         enddo
         ncstat=nf_put_vara_double(atpid, qa_id, startp, countp, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store wekta (Ekman velocity on T-grid) in 'wekt'
      if (outflat(4).eq.1) then
         start(1) = 1
         start(2) = 1
         start(3) = ntdone/noutat + 1
         count(1) = itwk
         count(2) = jtwk
         count(3) = 1
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (itwk,jtwk,wrk,wekta,nska)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jtwk
            do i=1,itwk
               wrk(i+itwk*(j-1)) = wekta(1+(i-1)*nska,1+(j-1)*nska)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat = nf_put_vara_double(attid, eka_id, start, count, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store interface height in 'h'
      if (outflat(5).eq.1) then
         startp(1) = 1
         startp(2) = 1
         startp(3) = 1
         startp(4) = ntdone/noutat + 1
         countp(1) = ipwk
         countp(2) = jpwk
         countp(3) = nla-1
         countp(4) = 1
         do k=1,nla-1
!$OMP       PARALLEL DO DEFAULT (NONE)
!$OMP&               PRIVATE (i,j)
!$OMP&               SHARED  (ipwk,jpwk,wrk,pa,nska,gpat,k)
!$OMP&               SCHEDULE (STATIC)
            do j=1,jpwk
               do i=1,ipwk
                  wrk(i+ipwk*(j-1)+ipwk*jpwk*(k-1)) =
     &              (  pa(1+(i-1)*nska,1+(j-1)*nska, k )
     &               - pa(1+(i-1)*nska,1+(j-1)*nska,k+1) )/gpat(k)
               enddo
            enddo
!$OMP       END PARALLEL DO
         enddo
         ncstat=nf_put_vara_double(atpid, ha_id, startp, countp, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  Store tauxa, tauya (dynamic windstress components) in 'taux' and 'tauy'
      if (outflat(6).eq.1) then
         start(1) = 1
         start(2) = 1
         start(3) = ntdone/noutat + 1
         count(1) = ipwk
         count(2) = jpwk
         count(3) = 1
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (ipwk,jpwk,wrk,tauxa,nska)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jpwk
            do i=1,ipwk
               wrk(i+ipwk*(j-1)) = tauxa(1+(i-1)*nska,1+(j-1)*nska)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat = nf_put_vara_double(atpid, txa_id, start, count, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (ipwk,jpwk,wrk,tauya,nska)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jpwk
            do i=1,ipwk
               wrk(i+ipwk*(j-1)) = tauya(1+(i-1)*nska,1+(j-1)*nska)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat = nf_put_vara_double(atpid, tya_id, start, count, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif

!!  hmixa in hmixa
      if (outflat(7).eq.1) then
         start(1) = 1
         start(2) = 1
         start(3) = ntdone/noutat + 1
         count(1) = itwk
         count(2) = jtwk
         count(3) = 1
!$OMP    PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j)
!$OMP&            SHARED  (itwk,jtwk,wrk,hmixa,nska)
!$OMP&            SCHEDULE (STATIC)
         do j=1,jtwk
            do i=1,itwk
               wrk(i+itwk*(j-1)) = hmixa(1+(i-1)*nska,1+(j-1)*nska)
            enddo
         enddo
!$OMP    END PARALLEL DO
         ncstat = nf_put_vara_double(attid, eta_id, start, count, wrk)
         if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      endif
#  endif /* not ocean_only */
#endif /* use_netcdf */

      END SUBROUTINE atnc_out

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE resave_nc
*
*     Write restart dump in netCDF format. All real fields are written in
*     NF_DOUBLE (double precision) format, to ensure restarts are exact.

*     Modules
      USE parameters
#ifdef use_netcdf
      USE atconst
      USE occonst
      USE atstate
      USE ocstate
      USE intrfac, ONLY : ast, astm, sst, sstm, hmixa, hmixam
#endif /* use_netcdf */
      USE timinfo, ONLY : tyrs

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments

#ifdef use_netcdf

*     Local parameters
      character (len=*), parameter :: subnam = 'resave_nc'

      integer i,j

!! netcdf variables
      integer ncstat,tdim,xopdim,yopdim,lodim,xotdim,yotdim,xapdim
      integer yapdim,ladim,xatdim,yatdim,xdims
      integer tim_id,xop_id,xot_id,yop_id
      integer yot_id,lo_id,xap_id,xat_id,yap_id,yat_id,la_id,dims(2)
      integer sst_id,sstm_id,po_id,pom_id,pdims(3),ast_id,astm_id
      integer hmixa_id,hmixam_id,pa_id,pam_id
      integer start(2),count(2),startp(3),countp(3)
      double precision xxo(nxpo),yyo(nypo),tmpo(nlo),
     &                 xxa(nxpa),yya(nypa),tmpa(nla)

!! now initialise netcdf file
      ncstat = nf_def_dim(restid, 'time', 1, tdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_dim(restid, 'xpo', nxpo, xopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'ypo', nypo, yopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'zo', nlo, lodim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_dim(restid, 'xto', nxto, xotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'yto', nyto, yotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_dim(restid, 'xpa', nxpa, xapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'ypa', nypa, yapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'za', nla, ladim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_dim(restid, 'xta', nxta, xatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(restid, 'yta', nyta, yatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time value
      xdims = tdim
      ncstat = nf_def_var(restid, 'time', NF_DOUBLE, 1, xdims, tim_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, tim_id, 'units', 5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xpo' which stores x gridpoints (p-grid)
      xdims = xopdim
      ncstat = nf_def_var(restid, 'xpo', NF_DOUBLE, 1, xdims, xop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, xop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xto' which stores x gridpoints (T-grid)
      xdims = xotdim
      ncstat = nf_def_var(restid, 'xto', NF_DOUBLE, 1, xdims, xot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, xot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'ypo' which stores y gridpoints (p-grid)
      xdims = yopdim
      ncstat = nf_def_var(restid, 'ypo', NF_DOUBLE, 1, xdims, yop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, yop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yto' which stores y gridpoints (T-grid)
      xdims = yotdim
      ncstat = nf_def_var(restid, 'yto', NF_DOUBLE, 1, xdims, yot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, yot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called
!!  'zo' to store ocean layer thicknesses
      xdims = lodim
      ncstat = nf_def_var(restid, 'zo', NF_DOUBLE, 1, xdims, lo_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, lo_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xpa' which stores x gridpoints (p-grid)
      xdims = xapdim
      ncstat = nf_def_var(restid, 'xpa', NF_DOUBLE, 1, xdims, xap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, xap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'xta' which stores x gridpoints (T-grid)
      xdims = xatdim
      ncstat = nf_def_var(restid, 'xta', NF_DOUBLE, 1, xdims, xat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, xat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'ypa' which stores y gridpoints (p-grid)
      xdims = yapdim
      ncstat = nf_def_var(restid, 'ypa', NF_DOUBLE, 1, xdims, yap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, yap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yta' which stores y gridpoints (T-grid)
      xdims = yatdim
      ncstat = nf_def_var(restid, 'yta', NF_DOUBLE, 1, xdims, yat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, yat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called
!!  'za' to store atmos. layer thicknesses
      xdims = ladim
      ncstat = nf_def_var(restid, 'za', NF_DOUBLE, 1, xdims, la_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, la_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'sst' to
!!  store ocean surface temperature - final timestep
      dims(1) = xotdim
      dims(2) = yotdim
      ncstat = nf_def_var(restid, 'sst', NF_DOUBLE, 2, dims, sst_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, sst_id, 'units', 1, 'K')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'sstm' to
!!  store ocean surface temperature - previous timestep
      dims(1) = xotdim
      dims(2) = yotdim
      ncstat = nf_def_var(restid, 'sstm', NF_DOUBLE, 2, dims, sstm_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, sstm_id, 'units', 1, 'K')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'po' to
!!  store ocean dynamic pressure - final timestep
      pdims(1) = xopdim
      pdims(2) = yopdim
      pdims(3) = lodim
      ncstat = nf_def_var(restid, 'po', NF_DOUBLE, 3, pdims, po_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, po_id, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'pom' to
!!  store ocean dynamic pressure - previous timestep
      pdims(1) = xopdim
      pdims(2) = yopdim
      pdims(3) = lodim
      ncstat = nf_def_var(restid, 'pom', NF_DOUBLE, 3, pdims, pom_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, pom_id, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'ast' to
!!  store atmos. surface temperature - final timestep
      dims(1) = xatdim
      dims(2) = yatdim
      ncstat = nf_def_var(restid, 'ast', NF_DOUBLE, 2, dims, ast_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, ast_id, 'units', 1, 'K')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'astm' to
!!  store atmos. surface temperature - previous timestep
      dims(1) = xatdim
      dims(2) = yatdim
      ncstat = nf_def_var(restid, 'astm', NF_DOUBLE, 2, dims, astm_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, astm_id, 'units', 1, 'K')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'hmixa' to
!!  store atmos. mixed layer thickness - final timestep
      dims(1) = xatdim
      dims(2) = yatdim
      ncstat = nf_def_var(restid, 'hmixa', NF_DOUBLE, 2,dims,hmixa_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, hmixa_id, 'units', 1, 'm')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 2-dimensional variable called 'hmixam' to
!!  store atmos. mixed layer thickness - previous timestep
      dims(1) = xatdim
      dims(2) = yatdim
      ncstat=nf_def_var(restid, 'hmixam', NF_DOUBLE, 2,dims,hmixam_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, hmixam_id, 'units', 1, 'm')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'pa' to
!!  store atmos. dynamic pressure - final timestep
      pdims(1) = xapdim
      pdims(2) = yapdim
      pdims(3) = ladim
      ncstat = nf_def_var(restid, 'pa', NF_DOUBLE, 3, pdims, pa_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, pa_id, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 3-dimensional variable called 'pam' to
!!  store atmos. dynamic pressure - previous timestep
      pdims(1) = xapdim
      pdims(2) = yapdim
      pdims(3) = ladim
      ncstat = nf_def_var(restid, 'pam', NF_DOUBLE, 3, pdims, pam_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(restid, pam_id, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Leave definition mode: entering data mode.
      ncstat = nf_enddef(restid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' resave_nc: enddef problem for restart file'
        call handle_err (ncstat, subnam)
      endif

!!  Write time value
      ncstat = nf_put_vara_double(restid, tim_id, 1, 1, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  OCEAN AXES
!!  Calculate x gridpoints and store in 'x' arrays.
!!  p-grid
      do i=1,nxpo
        xxo(i) = 1.0d-3*( xpo(i) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(restid, xop_id, 1, nxpo, xxo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid
      do i=1,nxto
        xxo(i) = 1.0d-3*( xto(i) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(restid, xot_id, 1, nxto, xxo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Calculate y gridpoints and store in 'y' arrays.
!!  p-grid
      do j=1,nypo
        yyo(j) = 1.0d-3*( ypo(j) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(restid, yop_id, 1, nypo, yyo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid
      do j=1,nyto
        yyo(j) = 1.0d-3*( yto(j) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(restid, yot_id, 1, nyto, yyo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert height into km and store in 'z'
      tmpo(1) = 0.5d-3*hoc(1)
      do i=2,nlo
        tmpo(i) = tmpo(i-1) + 0.5d-3*( hoc(i-1) + hoc(i) )
      enddo
      ncstat = nf_put_vara_double(restid, lo_id, 1, nlo, tmpo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  ATMOSPHERE AXES
!!  Calculate x gridpoints and store in 'x' arrays.
!!  p-grid
      do i=1,nxpa
        xxa(i) = 1.0d-3*xpa(i)
      enddo
      ncstat = nf_put_vara_double(restid, xap_id, 1, nxpa, xxa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid
      do i=1,nxta
        xxa(i) = 1.0d-3*xta(i)
      enddo
      ncstat = nf_put_vara_double(restid, xat_id, 1, nxta, xxa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Calculate y gridpoints and store in 'y' arrays.
!!  p-grid
      do j=1,nypa
        yya(j) = 1.0d-3*ypa(j)
      enddo
      ncstat = nf_put_vara_double(restid, yap_id, 1, nypa, yya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid
      do j=1,nyta
        yya(j) = 1.0d-3*yta(j)
      enddo
      ncstat = nf_put_vara_double(restid, yat_id, 1, nyta, yya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert height into km and store in 'z'
      tmpa(1) = 0.5d-3*hat(1)
      do i=2,nla
        tmpa(i) = tmpa(i-1) + 0.5d-3*( hat(i-1) + hat(i) )
      enddo
      ncstat = nf_put_vara_double(restid, la_id, 1, nla, tmpa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Now put ocean data
      start(1) = 1
      start(2) = 1
      count(1) = nxto
      count(2) = nyto
      ncstat = nf_put_vara_double(restid, sst_id, start, count, sst)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, sstm_id, start, count, sstm)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  ifndef atmos_only
      startp(1) = 1
      startp(2) = 1
      startp(3) = 1
      countp(1) = nxpo
      countp(2) = nypo
      countp(3) = nlo
      ncstat = nf_put_vara_double(restid, po_id, startp, countp, po)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, pom_id, startp, countp, pom)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  endif

!!  Now put atmos data
      start(1) = 1
      start(2) = 1
      count(1) = nxta
      count(2) = nyta
      ncstat = nf_put_vara_double(restid, ast_id, start, count, ast)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, astm_id, start, count, astm)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, hmixa_id, start, count, hmixa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, hmixam_id, start,count,hmixam)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  ifndef ocean_only
      startp(1) = 1
      startp(2) = 1
      startp(3) = 1
      countp(1) = nxpa
      countp(2) = nypa
      countp(3) = nla
      ncstat = nf_put_vara_double(restid, pa_id, startp, countp, pa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double(restid, pam_id, startp, countp, pam)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  endif

!! DONE
#endif /* use_netcdf */

      END SUBROUTINE resave_nc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE restart_nc
*
*     Read restart dump in netCDF format. All real fields are written in
*     NF_DOUBLE (double precision) format, to ensure restarts are exact.

*     Modules
      USE parameters
#ifdef use_netcdf
      USE atstate
      USE ocstate
      USE intrfac, ONLY : sst, sstm, ast, astm, hmixa, hmixam
#endif /* use_netcdf */
      USE timinfo, ONLY : tini

      IMPLICIT NONE

#ifdef use_netcdf
      INCLUDE 'netcdf.inc'
#endif /* use_netcdf */
*
*     Subroutine arguments

#ifdef use_netcdf

      integer i

*     Local parameters
      character (len=*), parameter :: subnam = 'restart_nc'

!!  netcdf variables
      integer ncstat,dim,varid

      print *,' '
      write (*,*) ' Initialise from netCDF restart file:'
      write (*,*) ' ------------------------------------'
!!  First check that all the dimensions are the correct length
      ncstat = nf_inq_dimid(restid, 'time', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.1) then
         print *,'netCDF restart file error'
         print *,'time dimension not equal to 1:'
         print *,i,1
         stop
      endif

      ncstat = nf_inq_dimid(restid, 'xpo', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nxpo) then
         print *,'netCDF restart file error'
         print *,'xpo dimension not equal to nxpo:'
         print *,i,nxpo
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'ypo', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nypo) then
         print *,'netCDF restart file error'
         print *,'ypo dimension not equal to nypo:'
         print *,i,nypo
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'zo', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nlo) then
         print *,'netCDF restart file error'
         print *,'zo dimension not equal to nlo:'
         print *,i,nlo
         stop
      endif

      ncstat = nf_inq_dimid(restid, 'xto', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nxto) then
         print *,'netCDF restart file error'
         print *,'xto dimension not equal to nxto:'
         print *,i,nxto
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'yto', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nyto) then
         print *,'netCDF restart file error'
         print *,'yto dimension not equal to nyto:'
         print *,i,nyto
         stop
      endif

      ncstat = nf_inq_dimid(restid, 'xpa', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nxpa) then
         print *,'netCDF restart file error'
         print *,'xpa dimension not equal to nxpa:'
         print *,i,nxpa
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'ypa', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nypa) then
         print *,'netCDF restart file error'
         print *,'ypa dimension not equal to nypa:'
         print *,i,nypa
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'za', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nla) then
         print *,'netCDF restart file error'
         print *,'za dimension not equal to nla:'
         print *,i,nla
         stop
      endif

      ncstat = nf_inq_dimid(restid, 'xta', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nxta) then
         print *,'netCDF restart file error'
         print *,'xta dimension not equal to nxta:'
         print *,i,nxta
         stop
      endif
      ncstat = nf_inq_dimid(restid, 'yta', dim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_inq_dimlen(restid, dim, i)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      if (i.ne.nyta) then
         print *,'netCDF restart file error'
         print *,'yta dimension not equal to nyta:'
         print *,i,nyta
         stop
      endif

!!  Checking is now done.
!!  Now read in all the variables.
!!  Each read involves finding the variable
!!  id (varid) and getting the data.
      ncstat = nf_inq_varid(restid, 'time', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, tini)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'sst', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, sst)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'sstm', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, sstm)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#  ifndef atmos_only
      ncstat = nf_inq_varid(restid, 'po', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, po)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'pom', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, pom)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  endif

      ncstat = nf_inq_varid(restid, 'ast', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, ast)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'astm', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, astm)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'hmixa', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, hmixa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'hmixam', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, hmixam)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#  ifndef ocean_only
      ncstat = nf_inq_varid(restid, 'pa', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, pa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(restid, 'pam', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(restid, varid, pam)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#  endif

!! DONE
      write (*,*) ' Initialisation from netCDF complete'
#endif /* use_netcdf */

      END SUBROUTINE restart_nc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#if defined (lu_mode) && (use_netcdf)

#  ifndef atmos_only

      SUBROUTINE oclu_init

*     Initialises the files oclu.nc

*     Modules
      USE parameters
      USE occonst
      USE timinfo, ONLY : noutstepoc
#    ifndef gmlu_param
      USE ocluvar, ONLY : aoxx,aoyy,aoxy
#    endif      

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'oclu_init'

!!  netCDF variables used locally
      integer ncstat, timodim, xopdim, yopdim, lodim
      integer xotdim, yotdim, odim, vdim4(4)
      integer xop_id, yop_id, xot_id, yot_id, lo_id
#    ifndef gmlu_param
      integer vdim3(3), start3(3), count3(3)
#    endif      

!!  Other variables used locally
      double precision xo(nxpo),yo(nypo),tmp(nlo)
      integer i

!!    Definition section: define dimensions and variables
!!    ===================================================
!!    Define four dimensions: x, y, z, time

!!    Dimension definitions for p-grid output file
!!    --------------------------------------------
      ncstat = nf_def_dim(ocrid, 'time', noutstepoc, timodim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    x and y dimensions for the p-grid
      ncstat = nf_def_dim(ocrid, 'xp', nxpo, xopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(ocrid, 'yp', nypo, yopdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  And here the z dimension
      ncstat = nf_def_dim(ocrid, 'z', nlo, lodim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Dimension definitions for T-grid output file
!!    --------------------------------------------

!!    x and y dimensions for the T-grid
      ncstat = nf_def_dim(ocrid, 'xt', nxto, xotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(ocrid, 'yt', nyto, yotdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for p-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xp' which stores x gridpoints (p-grid)
      odim = xopdim
!!      ncstat = nf_def_var(ocrid, 'xp', NF_DOUBLE, 1, odim, xop_id)
      ncstat = nf_def_var(ocrid, 'xp', NF_FLOAT, 1, odim, xop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, xop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, xop_id, 'long_name',
     &            21, 'Ocean X axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yp' which stores y gridpoints (p-grid)
      odim = yopdim
!!      ncstat = nf_def_var(ocrid, 'yp', NF_DOUBLE, 1, odim, yop_id)
      ncstat = nf_def_var(ocrid, 'yp', NF_FLOAT, 1, odim, yop_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, yop_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, yop_id, 'long_name',
     &            21, 'Ocean Y axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time data in years
      odim = timodim
!!      ncstat=nf_def_var(ocrid, 'time', NF_DOUBLE, 1, odim, timo_id)
      ncstat=nf_def_var(ocrid, 'time', NF_FLOAT, 1, odim, timo_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, timo_id, 'units',5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(ocrid,timo_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'z'
!!  to store layer thicknesses; p-grid only
      odim = lodim
!!      ncstat = nf_def_var(ocrid, 'z', NF_DOUBLE, 1, odim, lo_id)
      ncstat = nf_def_var(ocrid, 'z', NF_FLOAT, 1, odim, lo_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, lo_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, lo_id, 'long_name',
     &            26, 'Ocean mid-layer depth axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for T-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xt' which stores x gridpoints (T-grid)
      odim = xotdim
!!      ncstat = nf_def_var(ocrid, 'xt', NF_DOUBLE, 1, odim, xot_id)
      ncstat = nf_def_var(ocrid, 'xt', NF_FLOAT, 1, odim, xot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, xot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, xot_id, 'long_name',
     &            21, 'Ocean X axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yt' which stores y gridpoints (T-grid)
      odim = yotdim
!!      ncstat = nf_def_var(ocrid, 'yt', NF_DOUBLE, 1, odim, yot_id)
      ncstat = nf_def_var(ocrid, 'yt', NF_FLOAT, 1, odim, yot_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, yot_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, yot_id, 'long_name',
     &            21, 'Ocean Y axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define two 4-dimensional variables called 'ur' and 'vr' 
!!  to store random velocities
      vdim4(1) = xopdim
      vdim4(2) = yotdim
      vdim4(3) = lodim
      vdim4(4) = timodim
!!      ncstat = nf_def_var(ocrid, 'ur', NF_DOUBLE, 4, vdim4, uro_id)
      ncstat = nf_def_var(ocrid, 'ur', NF_FLOAT, 4, vdim4, uro_id)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining ur in ocean LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(ocrid, uro_id, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, uro_id, 'long_name',
     &              21, 'Ocean random velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      vdim4(1) = xotdim
      vdim4(2) = yopdim
      vdim4(3) = lodim
      vdim4(4) = timodim
!!      ncstat = nf_def_var(ocrid, 'vr', NF_DOUBLE, 4, vdim4, vro_id)
      ncstat = nf_def_var(ocrid, 'vr', NF_FLOAT, 4, vdim4, vro_id)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining vr in ocean LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(ocrid, vro_id, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, vro_id, 'long_name',
     &              21, 'Ocean random velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define three variables called 'axx', 'ayy' and 'axy' 
!!  to store variance tensor components
#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xopdim
      vdim4(2) = yotdim
      vdim4(3) = lodim
      vdim4(4) = timodim
!!      ncstat = nf_def_var(ocrid, 'axx', NF_DOUBLE, 4, vdim4, aoxx_id)
      ncstat = nf_def_var(ocrid, 'axx', NF_FLOAT, 4, vdim4, aoxx_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xopdim
      vdim3(2) = yotdim
      vdim3(3) = lodim
!!      ncstat = nf_def_var(ocrid, 'axx', NF_DOUBLE, 3, vdim3, aoxx_id)
      ncstat = nf_def_var(ocrid, 'axx', NF_FLOAT, 3, vdim3, aoxx_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining axx in ocean LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(ocrid, aoxx_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, aoxx_id, 'long_name',
     &                 21, 'Ocean variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xotdim
      vdim4(2) = yopdim
      vdim4(3) = lodim
      vdim4(4) = timodim
!!      ncstat = nf_def_var(ocrid, 'ayy', NF_DOUBLE, 4, vdim4, aoyy_id)
      ncstat = nf_def_var(ocrid, 'ayy', NF_FLOAT, 4, vdim4, aoyy_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xotdim
      vdim3(2) = yopdim
      vdim3(3) = lodim
!!      ncstat = nf_def_var(ocrid, 'ayy', NF_DOUBLE, 3, vdim3, aoyy_id)
      ncstat = nf_def_var(ocrid, 'ayy', NF_FLOAT, 3, vdim3, aoyy_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining ayy in ocean LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(ocrid, aoyy_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, aoyy_id, 'long_name',
     &                 21, 'Ocean variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xopdim
      vdim4(2) = yopdim
      vdim4(3) = lodim
      vdim4(4) = timodim
!!      ncstat = nf_def_var(ocrid, 'axy', NF_DOUBLE, 4, vdim4, aoxy_id)
      ncstat = nf_def_var(ocrid, 'axy', NF_FLOAT, 4, vdim4, aoxy_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xopdim
      vdim3(2) = yopdim
      vdim3(3) = lodim
!!      ncstat = nf_def_var(ocrid, 'axy', NF_DOUBLE, 3, vdim3, aoxy_id)
      ncstat = nf_def_var(ocrid, 'axy', NF_FLOAT, 3, vdim3, aoxy_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining axy in ocean LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(ocrid, aoxy_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(ocrid, aoxy_id, 'long_name',
     &                 21, 'Ocean variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Leave definition mode: entering data mode
      ncstat = nf_enddef(ocrid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' oclu_init: enddef problem for ocean LU file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif

!!  Calculate x gridpoints and store in 'x' arrays
!!  p-grid points
      do i=1,nxpo
        xo(i) = 1.0d-3*( xpo(i) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(ocrid, xop_id, 1, nxpo, xo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      do i=1,nxto
        xo(i) = 1.0d-3*( xto(i) - xpo(1) )
      enddo
      ncstat = nf_put_vara_double(ocrid, xot_id, 1, nxto, xo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Calculate y gridpoints and store in 'y' arrays
!!  p-grid points
      do i=1,nypo
        yo(i) = 1.0d-3*( ypo(i) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(ocrid, yop_id, 1, nypo, yo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      do i=1,nyto
        yo(i) = 1.0d-3*( yto(i) - ypo(1) )
      enddo
      ncstat = nf_put_vara_double(ocrid, yot_id, 1, nyto, yo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert mid-layer depths into km and store in 'z'
      tmp(1) = 0.5d-3*hoc(1)
      do i=2,nlo
        tmp(i) = tmp(i-1) + 0.5d-3*( hoc(i-1) + hoc(i) )
      enddo
      ncstat = nf_put_var_double(ocrid, lo_id, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Save variance tensor in the stationary case         
#    ifndef gmlu_param
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxpo
      count3(2) = nyto
      count3(3) = nlo
      ncstat = nf_put_vara_double(ocrid, aoxx_id, start3, count3, aoxx)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxto
      count3(2) = nypo
      count3(3) = nlo
      ncstat = nf_put_vara_double(ocrid, aoyy_id, start3, count3, aoyy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxpo
      count3(2) = nypo
      count3(3) = nlo
      ncstat = nf_put_vara_double(ocrid, aoxy_id, start3, count3, aoxy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif      

      END SUBROUTINE oclu_init

c     ==================================================================

      SUBROUTINE oclu_out

*     Puts data into the files oclu.nc

*     Modules
      USE parameters
      USE occonst
      USE ocluvar
      USE timinfo, ONLY : ntdone, noutoc, tyrs

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'oclu_out'

!!  netCDF variables used locally
      integer ncstat,startt,countt,start4(4),count4(4)

!!  Store current time as part of 'time' vector
      startt = ntdone/noutoc + 1
      countt = 1
      ncstat=nf_put_vara_double(ocrid, timo_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Store random velocities
      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutoc + 1
      count4(1) = nxpo
      count4(2) = nyto
      count4(3) = nlo
      count4(4) = 1
      ncstat = nf_put_vara_double(ocrid, uro_id, start4, count4, uro)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Store variance tensor in non-stationary case
#    ifdef gmlu_param      
      ncstat = nf_put_vara_double(ocrid, aoxx_id, start4, count4, aoxx)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutoc + 1
      count4(1) = nxto
      count4(2) = nypo
      count4(3) = nlo
      count4(4) = 1
      ncstat = nf_put_vara_double(ocrid, vro_id, start4, count4, vro)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_put_vara_double(ocrid, aoyy_id, start4, count4, aoyy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

#    ifdef gmlu_param      
      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutoc + 1
      count4(1) = nxpo
      count4(2) = nypo
      count4(3) = nlo
      count4(4) = 1
      ncstat = nf_put_vara_double(ocrid, aoxy_id, start4, count4, aoxy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

      END SUBROUTINE oclu_out

c     ==================================================================

      SUBROUTINE read_ocpod
    
*     Modules      
      USE ocluvar

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'
*
*     Subroutine arguments

*     Local parameters
      character (len=*), parameter :: subnam = 'read_ocpod_nc'
      integer ncstat,varid

!!TODO: Check dimension     

      print *,' '
      write (*,*) ' Read ocean POD from netCDF EOFs_noise file:'
      write (*,*) ' -------------------------------------------'

*     Read 2D variables 
      ncstat = nf_inq_varid(ocpodid, 'eofo', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat)
      ncstat = nf_get_var_double(ocpodid, varid, eofoc)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat)

*     Read 3D variables of diffusion tensor
      ncstat = nf_inq_varid(ocpodid, 'axxo', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_get_var_double(ocpodid, varid, apoxx)
#    else       
      ncstat = nf_get_var_double(ocpodid, varid, aoxx)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(ocpodid, 'ayyo', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_get_var_double(ocpodid, varid, apoyy)
#    else       
      ncstat = nf_get_var_double(ocpodid, varid, aoyy)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(ocpodid, 'axyo', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param
      ncstat = nf_get_var_double(ocpodid, varid, apoxy)
#    else       
      ncstat = nf_get_var_double(ocpodid, varid, aoxy)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!! DONE
      write (*,*) ' Read ocean POD from netCDF complete'
      
      END SUBROUTINE read_ocpod

#  endif /* not atmos_only */

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#  ifndef ocean_only

      SUBROUTINE atlu_init

*     Initialises the files atlu.nc

*     Modules
      USE parameters
      USE atconst
      USE timinfo, ONLY : noutstepat
#    ifndef gmlu_param
      USE atluvar, ONLY : aaxx,aayy,aaxy
#    endif      

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'atlu_init'

!!  netCDF variables used locally
      integer ncstat, timadim, xapdim, yapdim, ladim
      integer xatdim, yatdim, adim, vdim4(4)
      integer xap_id, yap_id, xat_id, yat_id, la_id
#    ifndef gmlu_param
      integer vdim3(3), start3(3), count3(3)
#    endif      

!!  Other variables used locally
      double precision xa(nxpa),ya(nypa),tmp(nla)
      integer i

!!    Definition section: define dimensions and variables
!!    ===================================================
!!    Define four dimensions: x, y, z, time

!!    Dimension definitions for p-grid output file
!!    --------------------------------------------
      ncstat = nf_def_dim(atrid, 'time', noutstepat, timadim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    x and y dimensions for the p-grid
      ncstat = nf_def_dim(atrid, 'xp', nxpa, xapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(atrid, 'yp', nypa, yapdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  And here the z dimension
      ncstat = nf_def_dim(atrid, 'z', nla, ladim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Dimension definitions for T-grid output file
!!    --------------------------------------------

!!    x and y dimensions for the T-grid
      ncstat = nf_def_dim(atrid, 'xt', nxta, xatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim(atrid, 'yt', nyta, yatdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for p-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xp' which stores x gridpoints (p-grid)
      adim = xapdim
!!      ncstat = nf_def_var(atrid, 'xp', NF_DOUBLE, 1, adim, xap_id)
      ncstat = nf_def_var(atrid, 'xp', NF_FLOAT, 1, adim, xap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, xap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, xap_id, 'long_name',
     &            21, 'Atmos X axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yp' which stores y gridpoints (p-grid)
      adim = yapdim
!!      ncstat = nf_def_var(atrid, 'yp', NF_DOUBLE, 1, adim, yap_id)
      ncstat = nf_def_var(atrid, 'yp', NF_FLOAT, 1, adim, yap_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, yap_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, yap_id, 'long_name',
     &            21, 'Atmos Y axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'time' which stores time data in years
      adim = timadim
!!      ncstat=nf_def_var(atrid, 'time', NF_DOUBLE, 1, adim, tima_id)
      ncstat=nf_def_var(atrid, 'time', NF_FLOAT, 1, adim, tima_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, tima_id, 'units',5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat=nf_put_att_text(atrid,tima_id,'long_name',9,'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a 1-dimensional variable called 'z'
!!  to store layer thicknesses; p-grid only
      adim = ladim
!!      ncstat = nf_def_var(atrid, 'z', NF_DOUBLE, 1, adim, la_id)
      ncstat = nf_def_var(atrid, 'z', NF_FLOAT, 1, adim, la_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, la_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, la_id, 'long_name',
     &            26, 'Atmos mid-layer depth axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!    Grid variable definitions for T-grid files
!!    ------------------------------------------
!!  Define a one-dimensional variable called
!!  'xt' which stores x gridpoints (T-grid)
      adim = xatdim
!!      ncstat = nf_def_var(atrid, 'xt', NF_DOUBLE, 1, adim, xat_id)
      ncstat = nf_def_var(atrid, 'xt', NF_FLOAT, 1, adim, xat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, xat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, xat_id, 'long_name',
     &            21, 'Atmos X axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define a one-dimensional variable called
!!  'yt' which stores y gridpoints (T-grid)
      adim = yatdim
!!      ncstat = nf_def_var(atrid, 'yt', NF_DOUBLE, 1, adim, yat_id)
      ncstat = nf_def_var(atrid, 'yt', NF_FLOAT, 1, adim, yat_id)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, yat_id, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, yat_id, 'long_name',
     &            21, 'Atmos Y axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define two 4-dimensional variables called 'ur' and 'vr' 
!!  to store random velocities
      vdim4(1) = xapdim
      vdim4(2) = yatdim
      vdim4(3) = ladim
      vdim4(4) = timadim
!!      ncstat = nf_def_var(atrid, 'ur', NF_DOUBLE, 4, vdim4, ura_id)
      ncstat = nf_def_var(atrid, 'ur', NF_FLOAT, 4, vdim4, ura_id)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining ur in atmos LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(atrid, ura_id, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, ura_id, 'long_name',
     &              21, 'Atmos random velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      vdim4(1) = xatdim
      vdim4(2) = yapdim
      vdim4(3) = ladim
      vdim4(4) = timadim
!!      ncstat = nf_def_var(atrid, 'vr', NF_DOUBLE, 4, vdim4, vra_id)
      ncstat = nf_def_var(atrid, 'vr', NF_FLOAT, 4, vdim4, vra_id)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining vr in atmos LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(atrid, vra_id, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, vra_id, 'long_name',
     &              21, 'Atmos random velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Define three variables called 'axx', 'ayy' and 'axy' 
!!  to store variance tensor components
#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xapdim
      vdim4(2) = yatdim
      vdim4(3) = ladim
      vdim4(4) = timadim
!!      ncstat = nf_def_var(atrid, 'axx', NF_DOUBLE, 4, vdim4, aaxx_id)
      ncstat = nf_def_var(atrid, 'axx', NF_FLOAT, 4, vdim4, aaxx_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xapdim
      vdim3(2) = yatdim
      vdim3(3) = ladim
!!      ncstat = nf_def_var(atrid, 'axx', NF_DOUBLE, 3, vdim3, aaxx_id)
      ncstat = nf_def_var(atrid, 'axx', NF_FLOAT, 3, vdim3, aaxx_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining axx in atmos LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(atrid, aaxx_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, aaxx_id, 'long_name',
     &                 21, 'Atmos variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xatdim
      vdim4(2) = yapdim
      vdim4(3) = ladim
      vdim4(4) = timadim
!!      ncstat = nf_def_var(atrid, 'ayy', NF_DOUBLE, 4, vdim4, aayy_id)
      ncstat = nf_def_var(atrid, 'ayy', NF_FLOAT, 4, vdim4, aayy_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xatdim
      vdim3(2) = yapdim
      vdim3(3) = ladim
!!      ncstat = nf_def_var(atrid, 'ayy', NF_DOUBLE, 3, vdim3, aayy_id)
      ncstat = nf_def_var(atrid, 'ayy', NF_FLOAT, 3, vdim3, aayy_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining ayy in atmos LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(atrid, aayy_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, aayy_id, 'long_name',
     &                 21, 'Atmos variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

#    ifdef gmlu_param      
*     Non-stationary noise, hence 4D arrays      
      vdim4(1) = xapdim
      vdim4(2) = yapdim
      vdim4(3) = ladim
      vdim4(4) = timadim
!!      ncstat = nf_def_var(atrid, 'axy', NF_DOUBLE, 4, vdim4, aaxy_id)
      ncstat = nf_def_var(atrid, 'axy', NF_FLOAT, 4, vdim4, aaxy_id)
#    else
*     Stationary noise, hence 3D arrays      
      vdim3(1) = xapdim
      vdim3(2) = yapdim
      vdim3(3) = ladim
!!      ncstat = nf_def_var(atrid, 'axy', NF_DOUBLE, 3, vdim3, aaxy_id)
      ncstat = nf_def_var(atrid, 'axy', NF_FLOAT, 3, vdim3, aaxy_id)
#    endif      
      if ( ncstat.ne.NF_NOERR ) then
        print *,' problem defining axy in atmos LU file'
        call handle_err (ncstat, subnam)
      endif
      ncstat = nf_put_att_text(atrid, aaxy_id, 'units', 5, 'm^2/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text(atrid, aaxy_id, 'long_name',
     &                 21, 'Atmos variance tensor')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Leave definition mode: entering data mode
      ncstat = nf_enddef(atrid)
      if ( ncstat.ne.NF_NOERR ) then
        print *,' atlu_init: enddef problem for atmos LU file'
        print *,' if problem is "One or more variable sizes'
        print *,' violate format constraints", either coarsen'
        print *,' spatial and/or temporal sampling of the data,'
        print *,' or create the file with "64-bit offset" format'
        call handle_err (ncstat, subnam)
      endif

!!  Calculate x gridpoints and store in 'x' arrays
!!  p-grid points
      do i=1,nxpa
        xa(i) = 1.0d-3*( xpa(i) - xpa(1) )
      enddo
      ncstat = nf_put_vara_double(atrid, xap_id, 1, nxpa, xa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      do i=1,nxta
        xa(i) = 1.0d-3*( xta(i) - xpa(1) )
      enddo
      ncstat = nf_put_vara_double(atrid, xat_id, 1, nxta, xa)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Calculate y gridpoints and store in 'y' arrays
!!  p-grid points
      do i=1,nypa
        ya(i) = 1.0d-3*( ypa(i) - ypa(1) )
      enddo
      ncstat = nf_put_vara_double(atrid, yap_id, 1, nypa, ya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  T-grid points
      do i=1,nyta
        ya(i) = 1.0d-3*( yta(i) - ypa(1) )
      enddo
      ncstat = nf_put_vara_double(atrid, yat_id, 1, nyta, ya)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Convert mid-layer depths into km and store in 'z'
      tmp(1) = 0.5d-3*hat(1)
      do i=2,nla
        tmp(i) = tmp(i-1) + 0.5d-3*( hat(i-1) + hat(i) )
      enddo
      ncstat = nf_put_var_double(atrid, la_id, tmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Save variance tensor in the stationary case         
#    ifndef gmlu_param
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxpa
      count3(2) = nyta
      count3(3) = nla
      ncstat = nf_put_vara_double(atrid, aaxx_id, start3, count3, aaxx)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxta
      count3(2) = nypa
      count3(3) = nla
      ncstat = nf_put_vara_double(atrid, aayy_id, start3, count3, aayy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      start3(1) = 1
      start3(2) = 1
      start3(3) = 1
      count3(1) = nxpa
      count3(2) = nypa
      count3(3) = nla
      ncstat = nf_put_vara_double(atrid, aaxy_id, start3, count3, aaxy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif      

      END SUBROUTINE atlu_init

c     ==================================================================

      SUBROUTINE atlu_out

*     Puts data into the files atlu.nc

*     Modules
      USE parameters
      USE atconst
      USE atluvar
      USE timinfo, ONLY : ntdone, noutat, tyrs

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'atlu_out'

!!  netCDF variables used locally
      integer ncstat,startt,countt,start4(4),count4(4)

!!  Store current time as part of 'time' vector
      startt = ntdone/noutat + 1
      countt = 1
      ncstat=nf_put_vara_double(atrid, tima_id, startt, countt, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!!  Store random velocities
      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutat + 1
      count4(1) = nxpa
      count4(2) = nyta
      count4(3) = nla
      count4(4) = 1
      ncstat = nf_put_vara_double(atrid, ura_id, start4, count4, ura)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
!!  Store variance tensor in non-stationary case
#    ifdef gmlu_param      
      ncstat = nf_put_vara_double(atrid, aaxx_id, start4, count4, aaxx)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutat + 1
      count4(1) = nxta
      count4(2) = nypa
      count4(3) = nla
      count4(4) = 1
      ncstat = nf_put_vara_double(atrid, vra_id, start4, count4, vra)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_put_vara_double(atrid, aayy_id, start4, count4, aayy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

#    ifdef gmlu_param      
      start4(1) = 1
      start4(2) = 1
      start4(3) = 1
      start4(4) = ntdone/noutat + 1
      count4(1) = nxpa
      count4(2) = nypa
      count4(3) = nla
      count4(4) = 1
      ncstat = nf_put_vara_double(atrid, aaxy_id, start4, count4, aaxy)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    endif

      END SUBROUTINE atlu_out

c     ==================================================================

      SUBROUTINE read_atpod
    
*     Modules      
      USE atluvar

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'
*
*     Subroutine arguments

*     Local parameters
      character (len=*), parameter :: subnam = 'read_atpod_nc'
      integer ncstat,varid

!!TODO: Check dimension     

      print *,' '
      write (*,*) ' Read atmos POD from netCDF EOFs_noise file:'
      write (*,*) ' -------------------------------------------'

*     Read 2D variables 
      ncstat = nf_inq_varid(atpodid, 'eofa', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat)
      ncstat = nf_get_var_double(atpodid, varid, eofat)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat)

*     Read 3D variables of diffusion tensor
      ncstat = nf_inq_varid(atpodid, 'axxa', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_get_var_double(atpodid, varid, apaxx)
#    else       
      ncstat = nf_get_var_double(atpodid, varid, aaxx)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(atpodid, 'ayya', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param      
      ncstat = nf_get_var_double(atpodid, varid, apayy)
#    else       
      ncstat = nf_get_var_double(atpodid, varid, aayy)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(atpodid, 'axya', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
#    ifdef gmlu_param
      ncstat = nf_get_var_double(atpodid, varid, apaxy)
#    else       
      ncstat = nf_get_var_double(atpodid, varid, aaxy)
#    endif
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!! DONE
      write (*,*) ' Read atmos POD from netCDF complete'
      
      END SUBROUTINE read_atpod

#  endif /* not ocean_only */
      
#endif /* (lu_mode) && (use_netcdf) */     
c
c***********************************************************************
c
#if defined (use_edvel) && (use_netcdf)

#  ifndef atmos_only
      SUBROUTINE read_ocedv
    
*     Modules      
      USE ocedvl

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'
*
*     Subroutine arguments

*     Local parameters
      character (len=*), parameter :: subnam = 'read_ocedv_nc'
      integer ncstat,varid

!!TODO: Check dimension     

      print *,' '
      write (*,*) ' Read ocean eddy velocities from netCDF file:'
      write (*,*) ' --------------------------------------------'

*     Read 3D variables
      ncstat = nf_inq_varid(ocedvid, 'uco', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)      
      ncstat = nf_get_var_double(ocedvid, varid, uoed)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(ocedvid, 'vco', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(ocedvid, varid, voed)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!! DONE
      write (*,*) ' Read ocean eddy velocities from netCDF complete'
      
      END SUBROUTINE read_ocedv
#  endif /* not atmos_only */

#  ifndef ocean_only
      SUBROUTINE read_atedv
    
*     Modules      
      USE atedvl

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'
*
*     Subroutine arguments

*     Local parameters
      character (len=*), parameter :: subnam = 'read_atedv_nc'
      integer ncstat,varid

!!TODO: Check dimension     

      print *,' '
      write (*,*) ' Read atmos eddy velocities from netCDF file:'
      write (*,*) ' --------------------------------------------'

*     Read 3D variables
      ncstat = nf_inq_varid(atedvid, 'uca', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)      
      ncstat = nf_get_var_double(atedvid, varid, uaed)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid(atedvid, 'vca', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var_double(atedvid, varid, vaed)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

!! DONE
      write (*,*) ' Read atmos eddy velocities from netCDF complete'
      
      END SUBROUTINE read_atedv
#  endif /* not ocean_only */

#endif /* (use_edvel) && (use_netcdf) */  
c
c***********************************************************************
*
      END MODULE nc_subs
*
c***********************************************************************
