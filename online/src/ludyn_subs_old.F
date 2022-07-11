c***********************************************************************
c     Q-GCM Version 1.5.0 : last modified 18/02/2013
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
      MODULE ludyn_subs

*     Contains generator of all random variables and updating the
*     variance tensor in Q-GCM_LU.
      
      IMPLICIT NONE
      PRIVATE
      
*     Subroutines
#ifdef lu_mode    
#  ifndef atmos_only      
      PUBLIC :: ocnoise
#  endif /* not atmos_only */      
#  ifndef ocean_only      
      PUBLIC :: atnoise
#  endif /* not ocean_only */      
      PRIVATE :: dvunif, dvnorm
#  ifdef gmlu_param
      PUBLIC :: sumtke
      PRIVATE :: bcomp
#    ifndef atmos_only      
      PRIVATE :: update_ocnoise
#    endif      
#    ifndef ocean_only      
      PRIVATE :: update_atnoise
#    endif      
#  endif /* gmlu_param */

      CONTAINS
#endif /* lu_mode */

c***********************************************************************
c
#if (defined lu_mode) && (!defined atmos_only)
          
      SUBROUTINE ocnoise
      
*     Generates the ocean random velocity using POD basis,
*     based on Karhuen-Loeve decomposition.

*     Modules
      USE parameters
      USE luparam
      USE ocluvar
#  ifdef gmlu_param      
      USE occonst, ONLY : amatoc
      USE ocstate, ONLY : po
#  endif

      IMPLICIT NONE

*     Local variables
      integer k,idr,ids,npu
      double precision rv(nmo),rtmp(neofo)
#  ifdef gmlu_param      
      double precision utmp(nxpo,nyto,nlo),vtmp(nxto,nypo,nlo),
     &                 bo(nxpo,nypo,nlo)
#  endif

*     Generate a Gaussian vector
      call dvnorm (nmo, rv)
      
!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (k,ids)
!$OMP&         SHARED  (ncore,nposub,rtmp,eofoc,rv)
*     Let each thread do a matrix-vector multiplication simutaneneously,
*     using the subroutine dgemv from LAPACK/BLAS      
!$OMP DO SCHEDULE (STATIC)      
      do k=1,ncore
        ids = 1 + (k-1)*nposub
        call dgemv ( 'N', nposub, nmo, 1.0d0, eofoc(ids,1), neofo,
     &                    rv(1), 1, 0.0d0, rtmp(ids), 1 )  
      enddo
!$OMP END DO NOWAIT
!$OMP END PARALLEL      

*     Using single thread to do for the residue points      
      if ( npores .ne. 0 ) then
        idr = 1 + ncore*nposub
        call dgemv ( 'N', npores, nmo, 1.0d0, eofoc(idr,1), neofo, 
     &                    rv(1), 1, 0.0d0, rtmp(idr), 1 )    
      endif

*     Reshape the large vector into 3D array 
      npu = nxpo*nyto*nlo
#  ifdef gmlu_param      
      utmp = reshape (rtmp(1:npu), (/nxpo,nyto,nlo/))
      vtmp = reshape (rtmp(npu+1:neofo), (/nxto,nypo,nlo/))
*     Compute ocean buoyancy (height fluctuations)      
      call bcomp (bo, po, amatoc, nxpo, nypo, nlo)
*     Update variables along isopycnal surface
      do k=1,nlo
        call update_ocnoise (uro(1,1,k), vro(1,1,k), aoxx(1,1,k),
     &                       aoyy(1,1,k), aoxy(1,1,k), bo(1,1,k), 
     &                       utmp(1,1,k), vtmp(1,1,k), apoxx(1,1,k),
     &                       apoyy(1,1,k), apoxy(1,1,k), keopod(k) )
      enddo
#  else
      uro = reshape (rtmp(1:npu), (/nxpo,nyto,nlo/))
      vro = reshape (rtmp(npu+1:neofo), (/nxto,nypo,nlo/))
#  endif

      END SUBROUTINE ocnoise

#endif
c***********************************************************************
c
#if (defined lu_mode) && (!defined ocean_only)
          
      SUBROUTINE atnoise
      
*     Generates the atmos random velocity using POD basis,
*     based on Karhuen-Loeve decomposition.

*     Modules
      USE parameters
      USE luparam
      USE atluvar
#  ifdef gmlu_param      
      USE atconst, ONLY : amatat
      USE atstate, ONLY : pa
#  endif

      IMPLICIT NONE

*     Local variables
      integer k,idr,ids,npu
      double precision rv(nma),rtmp(neofa)
#  ifdef gmlu_param      
      double precision utmp(nxpa,nyta,nla),vtmp(nxta,nypa,nla),
     &                 ba(nxpa,nypa,nla)
#  endif

*     Generate a Gaussian vector
      call dvnorm (nma, rv)
      
!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (k,ids)
!$OMP&         SHARED  (ncore,npasub,rtmp,eofat,rv)
*     Let each thread do a matrix-vector multiplication simutaneneously,
*     using the subroutine dgemv from LAPACK/BLAS      
!$OMP DO SCHEDULE (STATIC)      
      do k=1,ncore
        ids = 1 + (k-1)*npasub
        call dgemv ( 'N', npasub, nma, 1.0d0, eofat(ids,1), neofa,
     &                    rv(1), 1, 0.0d0, rtmp(ids), 1 )  
      enddo
!$OMP END DO NOWAIT
!$OMP END PARALLEL      

*     Using single thread to do for the residue points      
      if ( npares .ne. 0 ) then
        idr = 1 + ncore*npasub
        call dgemv ( 'N', npares, nma, 1.0d0, eofat(idr,1), neofa, 
     &                    rv(1), 1, 0.0d0, rtmp(idr), 1 )    
      endif

*     Reshape the large vector into 3D array 
      npu = nxpa*nyta*nla
#  ifdef gmlu_param      
      utmp = reshape (rtmp(1:npu), (/nxpa,nyta,nla/))
      vtmp = reshape (rtmp(npu+1:neofa), (/nxta,nypa,nla/))
*     Compute atmos buoyancy (height fluctuations)      
      call bcomp (ba, pa, amatat, nxpa, nypa, nla)
*     Update variables along isopycnal surface
      do k=1,nla
        call update_atnoise (ura(1,1,k), vra(1,1,k), aaxx(1,1,k),
     &                       aayy(1,1,k), aaxy(1,1,k), ba(1,1,k), 
     &                       utmp(1,1,k), vtmp(1,1,k), apaxx(1,1,k),
     &                       apayy(1,1,k), apaxy(1,1,k), keapod(k) )
      enddo
#  else
      ura = reshape (rtmp(1:npu), (/nxpa,nyta,nla/))
      vra = reshape (rtmp(npu+1:neofa), (/nxta,nypa,nla/))
#  endif

      END SUBROUTINE atnoise

#endif 
c      
c***********************************************************************
c
#if (!defined atmos_only) && (defined gmlu_param)

      SUBROUTINE update_ocnoise (u, v, axx, ayy, axy, b, uold, vold, 
     &                           axxo, ayyo, axyo, tke)

*     Update random velocity and variance tensor along isopycnal surface

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo
      USE occonst, ONLY : dxo,bccooc

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: u(nxpo,nyto),v(nxto,nypo),
     &                  axx(nxpo,nyto),ayy(nxto,nypo),axy(nxpo,nypo) 
      double precision, INTENT(IN) :: b(nxpo,nypo),uold(nxpo,nyto),
     &                  vold(nxto,nypo),axxo(nxpo,nyto), 
     &                  ayyo(nxto,nypo),axyo(nxpo,nypo),tke

*     Local variables
      integer i,j
      double precision dbdx(nxto,nypo),dbdy(nxpo,nyto),bx,by,up,vp,
     &                 dbfac,axxp,ayyp,axyp,zbfac,bcfac,bsfac,eps,
     &                 uke,vke,kefac 

*     Constant factors      
      zbfac = (1.0d0 - 0.5d0*bccooc)/(1.0d0 + 0.5d0*bccooc)
      bcfac = 1.0d0 + zbfac
      bsfac = 1.0d0 + zbfac**2
      eps = 1.0d16

*     Null boundaries for axy      
      axy = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,bx,by,up,vp,dbfac,axxp,ayyp,axyp)      
!$OMP&         SHARED  (b,dbdx,dbdy,uold,vold,axxo,ayyo,axyo,
!$OMP&                  dxo,bcfac,bsfac,eps,u,v,axx,ayy,axy)

*     Compute horizonal gradient of b
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxto
*       Southern boundary
        dbdx(i,1) = 0.0d0        
*       Inner points
        do j=2,nyto
          dbdx(i,j) = ( b(i+1,j) - b(i,j) )/dxo
        enddo
*       Northern boundary
        dbdx(i,nypo) = 0.0d0
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
      do j=1,nyto
*       Western boundary
        dbdy(1,j) = 0.0d0 
*       Inner points
        do i=2,nxto
          dbdy(i,j) = ( b(i,j+1) - b(i,j) )/dxo 
        enddo
*       Eastern boundary
        dbdy(nxpo,j) = 0.0d0
      enddo
!$OMP END DO

*     Projections on isopycnal surface     
*     ---------------- u and axx ---------------------     
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nyto
*       Western boundary (i=1)
        up = uold(1,j)
        vp = 0.25d0*bcfac*( vold(1,j+1) + vold(1,j) ) 
        axxp = axxo(1,j)
        ayyp = 0.25d0*bsfac*( ayyo(1,j+1) + ayyo(1,j) ) 
        axyp = 0.5d0*( axyo(1,j) + axyo(1,j+1) )
        bx = 0.25d0*bcfac*( dbdx(1,j+1) + dbdx(1,j) ) 
        by = dbdy(1,j)
        dbfac = eps*by/( eps*(bx**2 + by**2) ) 
        u(1,j) = dbfac*( by*up - bx*vp )
        axx(1,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                          bx*(bx*ayyp - by*axyp) )
*       Inner points (standard case)      
        do i=2,nxto
          up = uold(i,j)
          vp = 0.25d0*( vold(i-1,j+1) + vold(i-1,j) + 
     &                  vold(i,j+1) + vold(i,j) ) 
          axxp = axxo(i,j)
          ayyp = 0.25d0*( ayyo(i-1,j+1) + ayyo(i-1,j) + 
     &                    ayyo(i,j+1) + ayyo(i,j) ) 
          axyp = 0.5d0*( axyo(i,j) + axyo(i,j+1) )
          bx = 0.25d0*( dbdx(i-1,j+1) + dbdx(i-1,j) + 
     &                  dbdx(i,j+1) + dbdx(i,j) ) 
          by = dbdy(i,j)
          dbfac = eps*by/( eps*(bx**2 + by**2) ) 
          u(i,j) = dbfac*( by*up - bx*vp )
          axx(i,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                            bx*(bx*ayyp - by*axyp) )
        enddo
*       Eastern boundary (i=nxpo)
        up = uold(nxpo,j)
        vp = 0.25d0*bcfac*( vold(nxto,j+1) + vold(nxto,j) ) 
        axxp = axxo(nxpo,j)
        ayyp = 0.25d0*bsfac*( ayyo(nxto,j+1) + ayyo(nxto,j) ) 
        axyp = 0.5d0*( axyo(nxpo,j) + axyo(nxpo,j+1) )
        bx = 0.25d0*bcfac*( dbdx(nxto,j+1) + dbdx(nxto,j) ) 
        by = dbdy(nxpo,j)
        dbfac = eps*by/( eps*(bx**2 + by**2) ) 
        u(nxpo,j) = dbfac*( by*up - bx*vp )
        axx(nxpo,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                             bx*(bx*ayyp - by*axyp) )
      enddo
!$OMP END DO NOWAIT
*     ---------------- v and ayy ---------------------      
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxto
*       Southern boundary (j=1)      
        up = 0.25d0*bcfac*( uold(i,1) + uold(i+1,1) ) 
        vp = vold(i,1)
        axxp = 0.25d0*bsfac*( axxo(i,1) + axxo(i+1,1) ) 
        ayyp = ayyo(i,1)
        axyp = 0.5d0*( axyo(i,1) + axyo(i+1,1) )
        bx = dbdx(i,1)
        by = 0.25d0*bcfac*( dbdy(i,1) + dbdy(i+1,1) ) 
        dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
        v(i,1) = dbfac*( bx*vp - by*up )
        ayy(i,1) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                          bx*(bx*ayyp - by*axyp) )
*       Inner points (standard case)      
        do j=2,nyto
          up = 0.25d0*( uold(i,j) + uold(i,j-1) + 
     &                  uold(i+1,j-1) + uold(i+1,j) ) 
          vp = vold(i,j)
          axxp = 0.25d0*( axxo(i,j) + axxo(i,j-1) + 
     &                    axxo(i+1,j-1) + axxo(i+1,j) ) 
          ayyp = ayyo(i,j)
          axyp = 0.5d0*( axyo(i,j) + axyo(i+1,j) )
          bx = dbdx(i,j)
          by = 0.25d0*( dbdy(i,j) + dbdy(i,j-1) + 
     &                  dbdy(i+1,j-1) + dbdy(i+1,j) ) 
          dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
          v(i,j) = dbfac*( bx*vp - by*up )
          ayy(i,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                            bx*(bx*ayyp - by*axyp) )
        enddo
*       Northern boundary (j=nypo)        
        up = 0.25d0*bcfac*( uold(i,nyto) + uold(i+1,nyto) ) 
        vp = vold(i,nypo)
        axxp = 0.25d0*bsfac*( axxo(i,nyto) + axxo(i+1,nyto) ) 
        ayyp = ayyo(i,nypo)
        axyp = 0.5d0*( axyo(i,nypo) + axyo(i+1,nypo) )
        bx = dbdx(i,nypo)
        by = 0.25d0*bcfac*( dbdy(i,nyto) + dbdy(i+1,nyto) ) 
        dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
        v(i,nypo) = dbfac*( bx*vp - by*up )
        ayy(i,nypo) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                             bx*(bx*ayyp - by*axyp) )
      enddo
!$OMP END DO NOWAIT 
*     --------------------- axy ------------------------      
!$OMP DO SCHEDULE (STATIC)      
*     Only inner points      
      do j=2,nyto
        do i=2,nxto
          axxp = 0.5d0*( axxo(i,j-1) + axxo(i,j) )
          ayyp = 0.5d0*( ayyo(i-1,j) + ayyo(i,j) )
          axyp = axyo(i,j)
          bx = 0.5d0*( dbdx(i-1,j) + dbdx(i,j) )
          by = 0.5d0*( dbdy(i,j-1) + dbdy(i,j) )
          dbfac = - (eps**2)*bx*by/( (eps*(bx**2 + by**2))**2 ) 
          axy(i,j) = dbfac*( by*(by*axxp - bx*axyp) + 
     &                       bx*(bx*ayyp - by*axyp) )
        enddo
      enddo
!$OMP END DO NOWAIT
!$OMP END PARALLEL

*     Normalize by global TKE       
      call sumtke (axx, nxpo, nyto, 0.5d0, 1.0d0, uke)
      call sumtke (ayy, nxto, nypo, 1.0d0, 0.5d0, vke)
      kefac = tke/( uke + vke )
      u = sqrt(kefac)*u
      v = sqrt(kefac)*v
      axx = kefac*axx
      ayy = kefac*ayy
      axy = kefac*axy

      END SUBROUTINE update_ocnoise

#endif
c      
c***********************************************************************
c
#if (!defined ocean_only) && (defined gmlu_param)

      SUBROUTINE update_atnoise (u, v, axx, ayy, axy, b, uold, vold, 
     &                           axxo, ayyo, axyo, tke)

*     Update random velocity and variance tensor along isopycnal surface

*     Modules
      USE parameters, ONLY : nxta,nxpa,nyta,nypa
      USE atconst, ONLY : dxa,bccoat

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: u(nxpa,nyta),v(nxta,nypa),
     &                  axx(nxpa,nyta),ayy(nxta,nypa),axy(nxpa,nypa) 
      double precision, INTENT(IN) :: b(nxpa,nypa),uold(nxpa,nyta),
     &                  vold(nxta,nypa),axxo(nxpa,nyta), 
     &                  ayyo(nxta,nypa),axyo(nxpa,nypa),tke

*     Local variables
      integer i,j
      double precision dbdx(nxta,nypa),dbdy(nxpa,nyta),bx,by,up,vp,
     &                 dbfac,axxp,ayyp,axyp,zbfac,bcfac,bsfac,eps,
     &                 uke,vke,kefac 

*     Constant factors      
      zbfac = (1.0d0 - 0.5d0*bccoat)/(1.0d0 + 0.5d0*bccoat)
      bcfac = 1.0d0 + zbfac
      bsfac = 1.0d0 + zbfac**2
      eps = 1.0d16

*     Null boundaries for axy      
      axy = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,bx,by,up,vp,dbfac,axxp,ayyp,axyp)      
!$OMP&         SHARED  (b,dbdx,dbdy,uold,vold,axxo,ayyo,axyo,
!$OMP&                  dxa,bcfac,bsfac,eps,u,v,axx,ayy,axy)

*     Compute horizonal gradient of b
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxta
*       Southern boundary
        dbdx(i,1) = 0.0d0        
*       Inner points
        do j=2,nyta
          dbdx(i,j) = ( b(i+1,j) - b(i,j) )/dxa
        enddo
*       Northern boundary
        dbdx(i,nypa) = 0.0d0
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
      do j=1,nyta
*       Western boundary
        dbdy(1,j) = 0.0d0 
*       Inner points
        do i=2,nxta
          dbdy(i,j) = ( b(i,j+1) - b(i,j) )/dxa
        enddo
*       Eastern boundary
        dbdy(nxpa,j) = 0.0d0
      enddo
!$OMP END DO

*     Projections on isopycnal surface     
*     ---------------- u and axx ---------------------     
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nyta
*       Western boundary (i=1)
        up = uold(1,j)
        vp = 0.25d0*bcfac*( vold(1,j+1) + vold(1,j) ) 
        axxp = axxo(1,j)
        ayyp = 0.25d0*bsfac*( ayyo(1,j+1) + ayyo(1,j) ) 
        axyp = 0.5d0*( axyo(1,j) + axyo(1,j+1) )
        bx = 0.25d0*bcfac*( dbdx(1,j+1) + dbdx(1,j) ) 
        by = dbdy(1,j)
        dbfac = eps*by/( eps*(bx**2 + by**2) ) 
        u(1,j) = dbfac*( by*up - bx*vp )
        axx(1,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                          bx*(bx*ayyp - by*axyp) )
*       Inner points (standard case)      
        do i=2,nxta
          up = uold(i,j)
          vp = 0.25d0*( vold(i-1,j+1) + vold(i-1,j) + 
     &                  vold(i,j+1) + vold(i,j) ) 
          axxp = axxo(i,j)
          ayyp = 0.25d0*( ayyo(i-1,j+1) + ayyo(i-1,j) + 
     &                    ayyo(i,j+1) + ayyo(i,j) ) 
          axyp = 0.5d0*( axyo(i,j) + axyo(i,j+1) )
          bx = 0.25d0*( dbdx(i-1,j+1) + dbdx(i-1,j) + 
     &                  dbdx(i,j+1) + dbdx(i,j) ) 
          by = dbdy(i,j)
          dbfac = eps*by/( eps*(bx**2 + by**2) ) 
          u(i,j) = dbfac*( by*up - bx*vp )
          axx(i,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                            bx*(bx*ayyp - by*axyp) )
        enddo
*       Eastern boundary (i=nxpa)
        up = uold(nxpa,j)
        vp = 0.25d0*bcfac*( vold(nxta,j+1) + vold(nxta,j) ) 
        axxp = axxo(nxpa,j)
        ayyp = 0.25d0*bsfac*( ayyo(nxta,j+1) + ayyo(nxta,j) ) 
        axyp = 0.5d0*( axyo(nxpa,j) + axyo(nxpa,j+1) )
        bx = 0.25d0*bcfac*( dbdx(nxta,j+1) + dbdx(nxta,j) ) 
        by = dbdy(nxpa,j)
        dbfac = eps*by/( eps*(bx**2 + by**2) ) 
        u(nxpa,j) = dbfac*( by*up - bx*vp )
        axx(nxpa,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                             bx*(bx*ayyp - by*axyp) )
      enddo
!$OMP END DO NOWAIT
*     ---------------- v and ayy ---------------------      
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxta
*       Southern boundary (j=1)      
        up = 0.25d0*bcfac*( uold(i,1) + uold(i+1,1) ) 
        vp = vold(i,1)
        axxp = 0.25d0*bsfac*( axxo(i,1) + axxo(i+1,1) ) 
        ayyp = ayyo(i,1)
        axyp = 0.5d0*( axyo(i,1) + axyo(i+1,1) )
        bx = dbdx(i,1)
        by = 0.25d0*bcfac*( dbdy(i,1) + dbdy(i+1,1) ) 
        dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
        v(i,1) = dbfac*( bx*vp - by*up )
        ayy(i,1) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                          bx*(bx*ayyp - by*axyp) )
*       Inner points (standard case)      
        do j=2,nyta
          up = 0.25d0*( uold(i,j) + uold(i,j-1) + 
     &                  uold(i+1,j-1) + uold(i+1,j) ) 
          vp = vold(i,j)
          axxp = 0.25d0*( axxo(i,j) + axxo(i,j-1) + 
     &                    axxo(i+1,j-1) + axxo(i+1,j) ) 
          ayyp = ayyo(i,j)
          axyp = 0.5d0*( axyo(i,j) + axyo(i+1,j) )
          bx = dbdx(i,j)
          by = 0.25d0*( dbdy(i,j) + dbdy(i,j-1) + 
     &                  dbdy(i+1,j-1) + dbdy(i+1,j) ) 
          dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
          v(i,j) = dbfac*( bx*vp - by*up )
          ayy(i,j) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                            bx*(bx*ayyp - by*axyp) )
        enddo
*       Northern boundary (j=nypa)        
        up = 0.25d0*bcfac*( uold(i,nyta) + uold(i+1,nyta) ) 
        vp = vold(i,nypa)
        axxp = 0.25d0*bsfac*( axxo(i,nyta) + axxo(i+1,nyta) ) 
        ayyp = ayyo(i,nypa)
        axyp = 0.5d0*( axyo(i,nypa) + axyo(i+1,nypa) )
        bx = dbdx(i,nypa)
        by = 0.25d0*bcfac*( dbdy(i,nyta) + dbdy(i+1,nyta) ) 
        dbfac = eps*bx/( eps*(bx**2 + by**2) ) 
        v(i,nypa) = dbfac*( bx*vp - by*up )
        ayy(i,nypa) = (dbfac**2)*( by*(by*axxp - bx*axyp) + 
     &                             bx*(bx*ayyp - by*axyp) )
      enddo
!$OMP END DO NOWAIT 
*     --------------------- axy ------------------------      
!$OMP DO SCHEDULE (STATIC)      
*     Only inner points      
      do j=2,nyta
        do i=2,nxta
          axxp = 0.5d0*( axxo(i,j-1) + axxo(i,j) )
          ayyp = 0.5d0*( ayyo(i-1,j) + ayyo(i,j) )
          axyp = axyo(i,j)
          bx = 0.5d0*( dbdx(i-1,j) + dbdx(i,j) )
          by = 0.5d0*( dbdy(i,j-1) + dbdy(i,j) )
          dbfac = - (eps**2)*bx*by/( (eps*(bx**2 + by**2))**2 ) 
          axy(i,j) = dbfac*( by*(by*axxp - bx*axyp) + 
     &                       bx*(bx*ayyp - by*axyp) )
        enddo
      enddo
!$OMP END DO NOWAIT
!$OMP END PARALLEL

*     Normalize by global TKE       
      call sumtke (axx, nxpa, nyta, 0.5d0, 1.0d0, uke)
      call sumtke (ayy, nxta, nypa, 1.0d0, 0.5d0, vke)
      kefac = tke/( uke + vke )
      u = sqrt(kefac)*u
      v = sqrt(kefac)*v
      axx = kefac*axx
      ayy = kefac*ayy
      axy = kefac*axy

      END SUBROUTINE update_atnoise

#endif
c      
c***********************************************************************
c
#if defined (lu_mode) && (gmlu_param)

      SUBROUTINE bcomp (b, p, aaa, nxp, nyp, nl)

*     Compute buoyancy b from p at internal points.
*     Works for both atmosphere and ocean.

*     Input arguments:
*     p     : 3-D array of dynamic pressure at p points
*     aaa   : A(nl,nl) matrix linking pressures and eta
*     nxp   : x-dimension of p array
*     nyp   : y-dimension of p array
*     nl    : no. of QG layers

*     Output arguments:
*     b    : 3-D array of buoyancy at p points

*     Modules
      USE parameters

      IMPLICIT NONE

*     Subroutine arguments
      integer nxp,nyp,nl
      double precision b(nxp,nyp,nl),p(nxp,nyp,nl),aaa(nl,nl)
*
*     Local variables
      integer i,j,k


!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,k)
!$OMP&         SHARED  (nxp,nyp,nl,b,p,aaa)

*     k = 1 layer (top(ocean) or bottom(atmos.))
!$OMP DO SCHEDULE (STATIC)
      do j=1,nyp
        do i=1,nxp
          b(i,j,1) = -fnot*( aaa(1,1)*p(i,j,1) + aaa(1,2)*p(i,j,2) )
        enddo
      enddo
!$OMP END DO NOWAIT
*     Intermediate layers
      do k=2,nl-1
!$OMP   DO SCHEDULE (STATIC)
        do j=1,nyp
          do i=1,nxp
            b(i,j,k) = -fnot*( aaa(k,k-1)*p(i,j,k-1) + aaa(k,k)*p(i,j,k)
     &                        +aaa(k,k+1)*p(i,j,k+1) )
          enddo
        enddo
!$OMP   END DO NOWAIT
      enddo
*     k = nl layer (bottom(ocean) or top(atmos.))
!$OMP DO SCHEDULE (STATIC)
      do j=1,nyp
        do i=2,nxp-1
          b(i,j,nl) = -fnot*( aaa(nl,nl-1)*p(i,j,nl-1) 
     &                       +aaa(nl,nl)*p(i,j,nl) )
        enddo
      enddo
!$OMP END DO
!$OMP END PARALLEL

      END SUBROUTINE bcomp

#endif /* (lu_mode) && (gmlu_param) */
c      
c***********************************************************************
c
#if defined (lu_mode) && (gmlu_param)

      SUBROUTINE sumtke (val, nx, ny, facwe, facsn, answer)
*
*     Computes area integral of internal values of field
*     val(nx,ny). facwe controls contribution from western
*     & eastern boundaries; facsn controls contribution from
*     southern & northern boundaries. Integral returned as answer
*     Modified version with reduced accumulator error
*
*     N.B. Parallelised using "orphaned directives";
*     only intended for use in the parallel regions
*     established earlier in calling subroutine monnc_comp

      IMPLICIT NONE

*     Subroutine arguments
      integer nx,ny
      double precision val(nx,ny),facwe,facsn,answer
*
*     Local variables
      integer i,j
      double precision sumi,xxs,xxn
*
!$OMP SINGLE
      answer = 0.0d0
!$OMP END SINGLE
*     Inner points + facwe*(W & E boundaries)
!$OMP DO SCHEDULE (STATIC)
!$OMP&   PRIVATE (i,j,sumi)
!$OMP&   REDUCTION (+:answer)
      do j=2,ny-1
        sumi = facwe*val(1,j)
        do i=2,nx-1
          sumi = sumi + val(i,j)
        enddo
        sumi = sumi + facwe*val(nx,j)
        answer = answer + sumi
      enddo
!$OMP END DO

!$OMP SINGLE
*     N & S boundary contributions
      xxs = facwe*val(1, 1)
      xxn = facwe*val(1,ny)
*     Inner points
      do i=2,nx-1
        xxs = xxs + val(i, 1)
        xxn = xxn + val(i,ny)
      enddo
      xxs = xxs + facwe*val(nx, 1)
      xxn = xxn + facwe*val(nx,ny)

      answer = answer + facsn*( xxs + xxn )
!$OMP END SINGLE

      END SUBROUTINE sumtke

#endif /* (lu_mode) && (gmlu_param) */
c      
c***********************************************************************
c
#ifdef lu_mode

      SUBROUTINE dvnorm (n, x)         
     
*     Generate an array of double precision standard normal 
*     probability distribution.

      IMPLICIT NONE

*     I/O arguments
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
        call dvunif (2*m, r)
        x(xlid:xhid-1:2) = sqrt( -2.0d0*log(r(1:2*m-1:2)) ) *
     &                     cos( 2.0d0*PI*r(2:2*m:2) )
        x(xlid+1:xhid:2) = sqrt( -2.0d0*log(r(1:2*m-1:2)) ) *
     &                     sin( 2.0d0*PI*r(2:2*m:2) )
*     Case of odd number values
      else
        xhid = xhid - 1
        m = (xhid - xlid +1)/2 + 1
        call dvunif (2*m, r)
        x(xlid:xhid-1:2) = sqrt( -2.0d0*log(r(1:2*m-3:2)) ) *
     &                     cos( 2.0d0*PI*r(2:2*m-2:2) )
        x(xlid+1:xhid:2) = sqrt( -2.0d0*log(r(1:2*m-3:2)) ) *
     &                     sin( 2.0d0*PI*r(2:2*m-2:2) ) 
        x(n) = sqrt( -2.0d0*log(r(2*m-1)) ) * cos( 2.0d0*PI*r(2*m) )
      endif    

      END SUBROUTINE dvnorm

#endif /* lu_mode */
c      
c***********************************************************************
c
#ifdef lu_mode

      SUBROUTINE dvunif (n, r)       

*     Generate an array of double precision standard uniform 
*     probability distribution.

*     Modules
      USE luparam, ONLY : seed    

      IMPLICIT NONE

*     I/O arguments
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

#endif /* lu_mode */
c      
c***********************************************************************
c      
      END MODULE ludyn_subs
c
c***********************************************************************
