c     Test DMD noise
c
      PROGRAM test
      
      IMPLICIT NONE
      INCLUDE 'netcdf.inc'
      
*     Parameters
      integer  :: nxto,nyto,nlo,nxpo,nypo,nmco,nmro,nuo,nvo,nuvo
      PARAMETER ( nxto=48, nyto=60, nlo=3, nmco=14, nmro=56,
     &            nxpo=nxto+1, nypo=nyto+1, 
     &            nuo=nxpo*nyto*nlo, nvo=nxto*nypo*nlo, nuvo=nuo+nvo )

*     Local variables
      integer i,j,k,m,seed,ncore,nposub,npores,idr,ids,hnmr
      double precision xpo(nxpo),ypo(nxpo),zo(nlo),xto(nxto),yto(nxto),
     &                 dto,t,axyo(nxpo,nypo,nlo)
      double precision omgco(nmco),omgro(nmro),rgv(nmro) 
      double complex tcc(nmco),tcr(nmro),cgv(nmro),ctmp
      double precision, DIMENSION(nxpo,nyto,nlo,nmco) :: umdrco,umdico
      double precision, DIMENSION(nxto,nypo,nlo,nmco) :: vmdrco,vmdico
      double precision, DIMENSION(nxpo,nyto,nlo,nmro) :: umdrro,umdiro
      double precision, DIMENSION(nxto,nypo,nlo,nmro) :: vmdrro,vmdiro
      double complex dmdco(nuvo,nmco),dmdro(nuvo,nmro)
      double complex, DIMENSION(nuvo) :: wc,wr
      double precision, DIMENSION(nxpo,nyto,nlo) :: uco,uro,axxo
      double precision, DIMENSION(nxto,nypo,nlo) :: vco,vro,ayyo

*     NetCDF variables
      integer status,incid,varid,starts(4),counts(4),oncid,tdim,xpdim,
     &        ypdim,xtdim,ytdim,zdim,dims(4),tid,xpid,ypid,xtid,ytid,
     &        zid,ucid,vcid,urid,vrid,axxid,ayyid,axyid

*     OpenMP variables      
!$    integer nprocs, OMP_GET_NUM_PROCS, nthmax, OMP_GET_MAX_THREADS
!$    double precision OMP_GET_WTIME,stime1,etime1,ttime1

      print *
      print *, '*******************************************************'
      print *, '***            PROGRAM: Test of DMD Noise           ***'
      print *, '*******************************************************'

*     Examine OpenMP environment
!$    nprocs = OMP_GET_NUM_PROCS()
!$    nthmax = OMP_GET_MAX_THREADS()
!$    write(*,*) ' ' 
!$    write(*,*) ' OpenMP parallelism is activated'
!$    write(*,'(a,i5)') ' No. of processors available = ', nprocs
!$    write(*,'(a,i3)') ' Max. no. of threads available = ', nthmax

*     Initialize seed for random generator     
      call system_clock (count = seed)
      print *, 'Initial seed = ', seed
*     Distribute domain into subdomain by multi-threads
!$    ncore = nthmax 
      npores = mod(nuvo, ncore) ! nb. of residual points
      nposub = (nuvo - npores)/ncore ! nb. of subdomain points
      dto = 1440.0d0
      
      print *
      print *, 'Read DMD data'      
c     ------------------------------------------------------------------
c
*     Open netCDF file      
      status = nf_open ('ocludat.nc', nf_nowrite, incid)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     X-axis of p-grid (km)       
      status = nf_inq_varid (incid, 'xp', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, xpo)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Y-axis of p-grid (km)
      status = nf_inq_varid (incid, 'yp', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, ypo)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Z-axis (km)
      status = nf_inq_varid (incid, 'z', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, zo)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     X-axis of T-grid (km)       
      status = nf_inq_varid (incid, 'xt', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, xto)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Y-axis of T-grid (km)
      status = nf_inq_varid (incid, 'yt', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, yto)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Eigenvalues 
      status = nf_inq_varid (incid, 'omega_c', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, omgco)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      
      status = nf_inq_varid (incid, 'omega_r', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_var (incid, varid, omgro)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Eigenfunctions      
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = 1
      counts(1) = nxpo
      counts(2) = nyto
      counts(3) = nlo
      counts(4) = 1
      status = nf_inq_varid (incid, 'axx', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, axxo)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      counts(4) = nmco
      status = nf_inq_varid (incid, 'umode_real_c', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, umdrco)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_inq_varid (incid, 'umode_imag_c', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, umdico)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      counts(4) = nmro
      status = nf_inq_varid (incid, 'umode_real_r', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, umdrro)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_inq_varid (incid, 'umode_imag_r', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, umdiro)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      
      counts(1) = nxto
      counts(2) = nypo
      counts(4) = 1
      status = nf_inq_varid (incid, 'ayy', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, ayyo)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      counts(4) = nmco
      status = nf_inq_varid (incid, 'vmode_real_c', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, vmdrco)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_inq_varid (incid, 'vmode_imag_c', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, vmdico)
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      counts(4) = nmro
      status = nf_inq_varid (incid, 'vmode_real_r', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, vmdrro)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_inq_varid (incid, 'vmode_imag_r', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, vmdiro)
      if ( status.ne.NF_NOERR ) call handle_err (status)

      counts(1) = nxpo
      counts(4) = 1
      status = nf_inq_varid (incid, 'axy', varid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_get_vara_double (incid, varid, starts, counts, axyo)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Close file      
      status = nf_close(incid)
      if ( status.ne.NF_NOERR ) call handle_err (status) 
c      
c     ------------------------------------------------------------------

      print *
      print *, 'Complex and reshape DMD modes' 
c     ------------------------------------------------------------------
c
      dmdco(1:nuo,:) = dcmplx( reshape(umdrco, (/nuo,nmco/)),
     &                         reshape(umdico, (/nuo,nmco/)) ) 
      dmdco(nuo+1:nuvo,:) = dcmplx( reshape(vmdrco, (/nvo,nmco/)),
     &                              reshape(vmdico, (/nvo,nmco/)) )
      
      dmdro(1:nuo,:) = dcmplx( reshape(umdrro, (/nuo,nmro/)),
     &                         reshape(umdiro, (/nuo,nmro/)) ) 
      dmdro(nuo+1:nuvo,:) = dcmplx( reshape(vmdrro, (/nvo,nmro/)),
     &                              reshape(vmdiro, (/nvo,nmro/)) )
c      
c     ------------------------------------------------------------------

      print *
      print *, 'Generate correction drift and noise' 
c     ------------------------------------------------------------------
c
      t = 100000.0d0*dto

*     Generate real Gaussian vector
      call dvnorm (nmro, rgv, seed)
      hnmr = nmro/2

      wc = 0.0d0
      wr = 0.0d0
!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (m,k,ids,idr,ctmp)
!$OMP&         SHARED  (t,omgco,omgro,tcc,tcr,ncore,hnmr,rgv,
!$OMP&                  cgv,nposub,npores,dmdco,dmdro,wc,wr)

!$OMP DO SCHEDULE (STATIC)      
      do m=1,hnmr
        ctmp = dcmplx(rgv(m), rgv(m+hnmr))
        cgv(2*m-1) = ctmp 
        cgv(2*m) = dconjg(ctmp)
      enddo
!$OMP END DO NOWAIT

!$OMP DO SCHEDULE (STATIC)      
      do m=1,nmco
        tcc(m) = dcmplx( dcos(t*omgco(m)), dsin(t*omgco(m)) )
      enddo
!$OMP END DO NOWAIT

!$OMP DO SCHEDULE (STATIC)      
      do m=1,nmro
        tcr(m) = dcmplx( dcos(t*omgro(m)), dsin(t*omgro(m)) )*cgv(m)
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
      do k=1,ncore
        ids = 1 + (k-1)*nposub
        call zgemv ('N', nposub, nmco, 1.0d0, dmdco(ids,1), nuvo,
     &                   tcc(1), 1, 0.0d0, wc(ids), 1)  
        call zgemv ('N', nposub, nmro, 1.0d0, dmdro(ids,1), nuvo,
     &                   tcr(1), 1, 0.0d0, wr(ids), 1)  
      enddo  
!$OMP END DO
!$OMP END PARALLEL
      
      if ( npores .ne. 0 ) then
        idr = 1 + ncore*nposub
        call zgemv ('N', npores, nmco, 1.0d0, dmdco(idr,1), nuvo, 
     &                   tcc(1), 1, 0.0d0, wc(idr), 1)    
        call zgemv ('N', npores, nmro, 1.0d0, dmdro(idr,1), nuvo, 
     &                   tcr(1), 1, 0.0d0, wr(idr), 1)    
      endif

      uco = reshape (dble(wc(1:nuo)), (/nxpo,nyto,nlo/))
      vco = reshape (dble(wc(nuo+1:nuvo)), (/nxto,nypo,nlo/))      
      print *, 'Max value of uco = ', maxval(abs(uco))
      print *, 'Max value of vco = ', maxval(abs(vco))
  
      uro = reshape (dble(wr(1:nuo)), (/nxpo,nyto,nlo/))
      vro = reshape (dble(wr(nuo+1:nuvo)), (/nxto,nypo,nlo/))      
      print *, 'Max value of uro = ', maxval(abs(uro))
      print *, 'Max value of vro = ', maxval(abs(vro))
c      
c     ------------------------------------------------------------------

      print *
      print *, 'Write output file' 
c     ------------------------------------------------------------------
c
*     Create NetCDF file
      status = nf_create ('./test.nc', 
     &                    ior(nf_clobber,nf_64bit_offset), oncid)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Define dimensions
      status = nf_def_dim (oncid, 'time', 1, tdim)
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_def_dim (oncid, 'xp', nxpo, xpdim)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_def_dim (oncid, 'yp', nypo, ypdim)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_def_dim (oncid, 'z', nlo, zdim)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_def_dim (oncid, 'xt', nxto, xtdim)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_def_dim (oncid, 'yt', nyto, ytdim)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Define 1D variables with attributes
      status = nf_def_var(oncid, 'time', nf_double, 1, tdim, tid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, tid, 'units', 5, 'years')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, tid, 'long_name', 15, 
     &                         'Ocean time axis')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      
      status = nf_def_var(oncid, 'xp', nf_double, 1, xpdim, xpid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, xpid, 'units', 2, 'km')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, xpid, 'long_name', 21, 
     &                         'Ocean X axis (p-grid)')
      if ( status.ne.NF_NOERR ) call handle_err (status) 

      status = nf_def_var(oncid, 'yp', nf_double, 1, ypdim, ypid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, ypid, 'units', 2, 'km')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, ypid, 'long_name', 21, 
     &                         'Ocean Y axis (p-grid)')
      if ( status.ne.NF_NOERR ) call handle_err (status) 

      status = nf_def_var(oncid, 'z', nf_double, 1, zdim, zid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, zid, 'units', 2, 'km')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, zid, 'long_name', 26, 
     &                         'Ocean mid-layer depth axis')
      if ( status.ne.NF_NOERR ) call handle_err (status) 

      status = nf_def_var(oncid, 'xt', nf_double, 1, xtdim, xtid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, xtid, 'units', 2, 'km')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, xtid, 'long_name', 21, 
     &                         'Ocean X axis (T-grid)')
      if ( status.ne.NF_NOERR ) call handle_err (status) 

      status = nf_def_var(oncid, 'yt', nf_double, 1, ytdim, ytid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, ytid, 'units', 2, 'km')
      if ( status.ne.NF_NOERR ) call handle_err (status) 
      status = nf_put_att_text(oncid, ytid, 'long_name', 21, 
     &                         'Ocean Y axis (T-grid)')
      if ( status.ne.NF_NOERR ) call handle_err (status) 

*     Define 4D variable with attributes
      dims(1) = xpdim 
      dims(2) = ytdim
      dims(3) = zdim
      dims(4) = tdim
      status = nf_def_var(oncid, 'uc', nf_double, 4, dims, ucid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, ucid, 'units', 3, 'm/s')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, ucid, 'long_name', 28,
     &                         'Ocean zonal correction drift')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      
      status = nf_def_var(oncid, 'ur', nf_double, 4, dims, urid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, urid, 'units', 3, 'm/s')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, urid, 'long_name', 26,
     &                         'Ocean zonal velocity noise')
      if ( status.ne.NF_NOERR ) call handle_err (status)

      dims(1) = xtdim 
      dims(2) = ypdim
      dims(3) = zdim
      dims(4) = tdim
      status = nf_def_var(oncid, 'vc', nf_double, 4, dims, vcid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, vcid, 'units', 3, 'm/s')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, vcid, 'long_name', 33,
     &                         'Ocean meridional correction drift')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      
      status = nf_def_var(oncid, 'vr', nf_double, 4, dims, vrid)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, vrid, 'units', 3, 'm/s')
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_att_text(oncid, vrid, 'long_name', 31,
     &                         'Ocean meridional velocity noise')
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Leave definition mode and entering data mode
      status = nf_enddef(oncid)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Write 1D data to variables
      status = nf_put_vara_double(oncid, tid, 1, 1, t/365.0d0/86400.0d0)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, xpid, 1, nxpo, xpo)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, ypid, 1, nypo, ypo)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, zid, 1, nlo, zo)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, xtid, 1, nxto, xto)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, ytid, 1, nyto, yto)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Write 4D data to variables
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = 1
      counts(1) = nxpo
      counts(2) = nyto
      counts(3) = nlo
      counts(4) = 1
      status = nf_put_vara_double(oncid, ucid, starts, counts, uco)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, urid, starts, counts, uro)
      if ( status.ne.NF_NOERR ) call handle_err (status)

      counts(1) = nxto
      counts(2) = nypo
      status = nf_put_vara_double(oncid, vcid, starts, counts, vco)
      if ( status.ne.NF_NOERR ) call handle_err (status)
      status = nf_put_vara_double(oncid, vrid, starts, counts, vro)
      if ( status.ne.NF_NOERR ) call handle_err (status)

*     Save netCDF file and close it
      status = nf_close(oncid)
      if ( status.ne.NF_NOERR ) call handle_err (status)

      END PROGRAM test
c
c**************************************************************************
c      
      SUBROUTINE dvnorm (n, x, seed)         
     
*     Generate an array of double precision standard normal 
*     probability distribution.

      IMPLICIT NONE

*     I/O arguments
      integer seed
      integer, INTENT(IN) :: n
      double precision, INTENT(OUT) :: x(n)

*     Local variables
      integer m,xlid,xhid
      double precision r(n+1)
      DOUBLE PRECISION PI
      PARAMETER ( PI=3.14159265358979324D0 )

*     x(n) is a sample of n standard normal PDF.
*     r(n+1) is used to store n+1 uniform random values.
*     xlid, xhid records the range of entries of x that need to compute.      

*     Record the range of x we need to fill in.
      xlid = 1
      xhid = n

*     Case of single value
      if ( xhid-xlid+1 .eq. 1 ) then
!!        call dunif (r(1)) 
        if ( r(1) .eq. 0.0d0 ) then
          print *
          print *, 'dvec_normal: Fatal error!'
          print *, 'dvec_normal: Returns a value of zero.'
          stop
        endif
*     Case of even number values
      elseif ( mod(xhid-xlid,2) .eq. 1 ) then
        m = (xhid - xlid + 1)/2
        call dvunif (2*m, r, seed)
        x(xlid:xhid-1:2) = sqrt( -2.0d0*log(r(1:2*m-1:2)) ) *
     &                     cos( 2.0d0*PI*r(2:2*m:2) )
        x(xlid+1:xhid:2) = sqrt( -2.0d0*log(r(1:2*m-1:2)) ) *
     &                     sin( 2.0d0*PI*r(2:2*m:2) )
*     Case of odd number values
      else
        xhid = xhid - 1
        m = (xhid - xlid +1)/2 + 1
        call dvunif (2*m, r, seed)
        x(xlid:xhid-1:2) = sqrt( -2.0d0*log(r(1:2*m-3:2)) ) *
     &                     cos( 2.0d0*PI*r(2:2*m-2:2) )
        x(xlid+1:xhid:2) = sqrt( -2.0d0*log(r(1:2*m-3:2)) ) *
     &                     sin( 2.0d0*PI*r(2:2*m-2:2) ) 
        x(n) = sqrt( -2.0d0*log(r(2*m-1)) ) * cos( 2.0d0*PI*r(2*m) )
      endif    

      END SUBROUTINE dvnorm
c      
c***********************************************************************
c
      SUBROUTINE dvunif (n, r, seed)       

*     Generate an array of double precision standard uniform 
*     probability distribution.

      IMPLICIT NONE

*     I/O arguments
      integer seed
      integer, INTENT(IN) :: n
      double precision, INTENT(OUT) :: r(n)

*     Local variables
      integer i,k

!!TODO: Use OMP or random_number      
      do i=1,n
        k = seed / 127773
        seed = 16807 * (seed - k*127773) - k*2836
        if ( seed < 0 ) then
          seed = seed + 2147483647
        endif
        r(i) = dble(seed) * 4.656612875d-10
      enddo

      END SUBROUTINE dvunif
c      
c***********************************************************************
c      
      SUBROUTINE handle_err (ncstat)

*     Handle source of errors based on netCDF states
      
      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     I/O arguments
      integer, INTENT(IN) :: ncstat

*     Routine which interprets errors from netCDF output functions,
*     prints them to standard output and then kills the whole run.
      if ( ncstat.ne.NF_NOERR ) then
        print *, trim( nf_strerror(ncstat) )
        stop 'netCDF:: STOPPED'
      endif

      END SUBROUTINE handle_err
