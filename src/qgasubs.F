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
      MODULE qgasubs

*     Contains subroutines qgastep and atadif which step
*     the atmospheric vorticity evolution equation (7.14).

*     Modules

      IMPLICIT NONE

      PRIVATE

#ifndef ocean_only
      PUBLIC  :: qgastep
      PRIVATE :: atadif
#  ifdef lu_mode      
      PRIVATE :: atadslu
#  endif
#  ifdef use_edvel
      PRIVATE :: ataded
#  endif      

      CONTAINS
#endif

c***********************************************************************
c
#ifndef ocean_only

      SUBROUTINE qgastep
*
*     Steps the evolution equation (7.14) for
*     the atmospheric vorticity. Leapfrog timestep.

*     Modules
      USE parameters
      USE atconst
      USE athomog
      USE atstate
#  ifdef lu_mode      
      USE atluvar
#  endif
#  ifdef use_edvel
      USE atedvl
#  endif      

      IMPLICIT NONE

*     Subroutine arguments
*
*     Local variables
      integer i,j,k
      double precision adfaca,zbfaca,fohfac(nla),del2p(nxpa,nypa),
     &                 dqdt(nxpa,nypa,nla),qdot(nla),qold,
     &                 dqdtlu(nxpa,nypa,nla),
     &                 dqdted(nxpa,nypa,nla)

      adfaca = 1.0d0/(12.0d0*dxa*dya*fnot)
*     Version with nondimensional bccoat
      zbfaca = bccoat*dxam2/( 0.5d0*bccoat + 1.0d0 )
      do k=1,nla
        fohfac(k) = fnot/hat(k)
      enddo

*     ATMOSPHERIC q STEP (INTERIOR)

      do k=1,nla

*       Compute Del-sqd(p) at previous time level for dissipation
*       ---------------------------------------------------------
!$OMP   PARALLEL DO DEFAULT (NONE)
!$OMP&           PRIVATE (i,j)
!$OMP&           SHARED  (k,del2p,pam,dxam2)
!$OMP&           SCHEDULE (STATIC)
        do j=2,nypa-1
*         Western boundary (periodic)
          del2p( 1 ,j) = (  pam(1,j-1,k) + pam(nxpa-1,j,k) + pam(2,j,k)
     &                    + pam(1,j+1,k) - 4.0d0*pam(1,j,k) )*dxam2
*         Inner points (standard case)
          do i=2,nxpa-1
            del2p(i,j) = (  pam(i,j-1,k) + pam(i-1,j,k) + pam(i+1,j,k)
     &                    + pam(i,j+1,k) - 4.0d0*pam(i,j,k) )*dxam2
          enddo
*         Eastern boundary (periodic)
          del2p(nxpa,j) = del2p( 1 ,j)
        enddo  !meridional boundaries (excluding corners)
!$OMP   END PARALLEL DO

*       Zonal boundaries (including corners) - first order differencing
*       No x-derivative term because p is constant along boundary
        do i=1,nxpa
          del2p(i,  1 ) = zbfaca*( pam(i,   2  ,k) - pam(i,  1 ,k) )
          del2p(i,nypa) = zbfaca*( pam(i,nypa-1,k) - pam(i,nypa,k) )
        enddo

*       Further differentiate contents of del2p to get Del-6th(lagged p).
*       Initialise dqdt in each layer with (Jacobian)
*       advective term plus Del-6th frictional term.
*       Include (1/fnot) factor in multiplicative factor adfaca
        call atadif (dqdt(1,1,k), del2p(1,1), ah4at(k), zbfaca,
     &               ap5sat(k), ap5nat(k), pa(1,1,k), qa(1,1,k),
     &               adfaca, ajisat(k), ajinat(k))

#  ifdef lu_mode
        call atadslu (dqdtlu(1,1,k), qa(1,1,k), pa(1,1,k), 
     &                ura(1,1,k), vra(1,1,k), 
     &                aaxx(1,1,k), aayy(1,1,k), aaxy(1,1,k))
#  else
        dqdtlu = 0.0d0
#  endif /* lu_mode */

!!#  ifdef use_edvel        
!!        call ataded (dqdted(1,1,k), qa(1,1,k), 
!!     &               uaed(1,1,k), vaed(1,1,k))
!!#  else
        dqdted = 0.0d0
!!#  endif /* use_edvel */

      enddo

*     Specify forcing and mass correction; then timestep
*     --------------------------------------------------
*     These are the layer-specific terms near the top and bottom
!$OMP PARALLEL DO DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,k,qdot,qold)
!$OMP&         SHARED  (dqdt,fohfac,entat,wekpa,qa,qam,tdta,
!$OMP&                  dta,dqdtlu,dqdted)
!$OMP&         SCHEDULE (STATIC)
      do j=2,nypa-1
        do i=1,nxpa
          qdot( 1 ) = dqdt(i,j,1)
     &               + fohfac(1)*( entat(i,j) - wekpa(i,j) )
          qdot( 2 ) = dqdt(i,j,2) - fohfac(2)*entat(i,j)
          do k=3,nla
            qdot(k) = dqdt(i,j,k)
          enddo
*         Step the values of qa except at zonal boundaries
*         The boundary values of qa will be updated later by atqzbd
          do k=1,nla
            qold = qa(i,j,k)
            qa(i,j,k) = qam(i,j,k) + tdta*qdot(k)
     &                             + dta*dqdtlu(i,j,k)
     &                             + tdta*dqdted(i,j,k)       
            qam(i,j,k) = qold
          enddo
        enddo
      enddo
!$OMP END PARALLEL DO

*     Copy zonal boundary values, which have not been updated
*     Boundary points are all updated later by atqzbd
      do k=1,nla
        do i=1,nxpa
          qam(i,  1 ,k) = qa(i,  1 ,k)
          qam(i,nypa,k) = qa(i,nypa,k)
        enddo
      enddo

      END SUBROUTINE qgastep

#endif /* not ocean_only */
c
c***********************************************************************
c
#ifndef ocean_only

      SUBROUTINE atadif (dqdt, d2p, ah4atk, zbfaca, ahp5is, ahp5in,
     &                   p, q, adfaca, ajisat, ajinat)
*
*     Computes advective + diffusive contributions to evolution
*     equation for q within an atmospheric layer. Computes friction
*     term Del-6th(p). Initialises array dqdt with adfaca times
*     Jacobian advection term J(q,p) minus (ah4atk/fnot)*Del-6th(p).
*     Uses Arakawa energy and enstrophy conserving 9-point Jacobian
*     Also computes the near-boundary contributions to the momentum
*     constraint equations: Ah4*Int ( d5p/dy5 ) dx and Int ( Jacobian )
*     Array "d2p" contains on entry Del-sqd(time-lagged p),
*     and is unchanged on exit.

*     Modules
      USE parameters
      USE atconst

      IMPLICIT NONE

*     Subroutine arguments
      double precision dqdt(nxpa,nypa),d2p(nxpa,nypa),ah4atk,zbfaca,
     &                 ahp5is,ahp5in,p(nxpa,nypa),q(nxpa,nypa),
     &                 adfaca,ajisat,ajinat
*
*     Local variables
      integer i,j
      double precision d4p(nxpa,nypa),d6p,ah4fac,ah5sms,ah5smn,
     &                 aj5sms,aj5smn,aj9sms,aj9smn

      ah4fac = ah4atk/fnot

*     Southern boundary Jacobian term
*     -------------------------------
*     Cyclic advection at Western boundary.
      aj5sms = 0.5d0*q(  1 , 1 )*( p( 2 , 2 )-p(nxpa-1, 2 ) )
      aj9sms = 0.5d0*q(  1 , 2 )*( p( 2 , 2 )-p(nxpa-1, 2 ) )
*     Internal points
      do i=2,nxpa-1
        aj5sms = aj5sms + q(i, 1 )*( p(i+1, 2 )-p(i-1, 2 ) )
        aj9sms = aj9sms + q(i, 2 )*( p(i+1, 2 )-p(i-1, 2 ) )
      enddo
*     Cyclic advection at Eastern boundary.
      aj5sms = aj5sms + 0.5d0*q(nxpa, 1 )*( p( 2 , 2 )-p(nxpa-1, 2 ) )
      aj9sms = aj9sms + 0.5d0*q(nxpa, 2 )*( p( 2 , 2 )-p(nxpa-1, 2 ) )
      ajisat = fnot*adfaca*( aj5sms + 2.0d0*aj9sms )
*     Convert boundary constraint sum to integral
      ajisat = dxa*dya*ajisat
*
*     COMPUTE HORIZONTAL FRICTION - ATMOSPHERE
*
*     Compute Del-4th(p)
*     ------------------
*     Atmosphere is periodic E-W.
*
*     Apply N & S boundary condition: mixed
      do i=1,nxpa
        d4p(i,  1 ) = zbfaca*( d2p(i,   2  ) - d2p(i,  1 ) )
        d4p(i,nypa) = zbfaca*( d2p(i,nypa-1) - d2p(i,nypa) )
      enddo

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,d6p)
!$OMP&         SHARED  (d4p,dxam2,d2p,dqdt,ah4fac,adfaca,p,q)

*     Compute Del-4th(p) except at N & S boundaries
!$OMP DO SCHEDULE (STATIC)
      do j=2,nypa-1
*       Western boundary (periodic)
        d4p(  1 ,j) = dxam2*(  d2p(1,j-1) + d2p(nxpa-1,j) + d2p(2,j)
     &                       + d2p(1,j+1) - 4.0d0*d2p(1,j) )
*       Inner points (standard case)
        do i=2,nxpa-1
          d4p(i,j) = dxam2*(  d2p(i,j-1) + d2p(i-1,j) + d2p(i+1,j)
     &                      + d2p(i,j+1) - 4.0d0*d2p(i,j) )
        enddo
*       Eastern boundary (periodic)
        d4p(nxpa,j) = d4p(  1 ,j)
      enddo
!$OMP END DO
*
*     Compute advective (Jacobian) and diffusive contributions to dq/dt
*     -----------------------------------------------------------------
*     Compute Del-6th(p) except at N & S boundaries
*     No need to apply N & S boundary conditions
*     Initialise dq/dt with Jacobian advection and diffusive terms
*     Atmosphere is periodic E-W.
!$OMP DO SCHEDULE (STATIC)
      do j=2,nypa-1
*       Western boundary (i=1) (periodic)
        d6p = dxam2*(  d4p(1,j-1) + d4p(nxpa-1,j) + d4p(2,j)
     &               + d4p(1,j+1) - 4.0d0*d4p(1,j) )
*       Cyclic advection at Western boundary.
        dqdt(1,j) = adfaca*
     &             ( (q( 2 , j )-q(nxpa-1, j ))*(p( 1 ,j+1)-p(  1 ,j-1))
     &              +(q( 1 ,j-1)-q(  1 ,j+1))*(p( 2 , j )-p(nxpa-1, j ))
     &              + q(   2  , j )*( p(   2  ,j+1) - p(   2  ,j-1) )
     &              - q(nxpa-1, j )*( p(nxpa-1,j+1) - p(nxpa-1,j-1) )
     &              - q(   1  ,j+1)*( p(   2  ,j+1) - p(nxpa-1,j+1) )
     &              + q(   1  ,j-1)*( p(   2  ,j-1) - p(nxpa-1,j-1) )
     &              + p(   1  ,j+1)*( q(   2  ,j+1) - q(nxpa-1,j+1) )
     &              - p(   1  ,j-1)*( q(   2  ,j-1) - q(nxpa-1,j-1) )
     &              - p(   2  , j )*( q(   2  ,j+1) - q(   2  ,j-1) )
     &              + p(nxpa-1, j )*( q(nxpa-1,j+1) - q(nxpa-1,j-1) ) )
     &              - ah4fac*d6p
*       Inner points (standard case)
        do i=2,nxpa-1
          d6p = dxam2*(  d4p(i,j-1) + d4p(i-1,j) + d4p(i+1,j)
     &                 + d4p(i,j+1) - 4.0d0*d4p(i,j) )
          dqdt(i,j) = adfaca*
     &               ( (q(i+1, j )-q(i-1, j ))*(p( i ,j+1)-p( i ,j-1))
     &                +(q( i ,j-1)-q( i ,j+1))*(p(i+1, j )-p(i-1, j ))
     &                + q(i+1, j )*( p(i+1,j+1) - p(i+1,j-1) )
     &                - q(i-1, j )*( p(i-1,j+1) - p(i-1,j-1) )
     &                - q( i ,j+1)*( p(i+1,j+1) - p(i-1,j+1) )
     &                + q( i ,j-1)*( p(i+1,j-1) - p(i-1,j-1) )
     &                + p( i ,j+1)*( q(i+1,j+1) - q(i-1,j+1) )
     &                - p( i ,j-1)*( q(i+1,j-1) - q(i-1,j-1) )
     &                - p(i+1, j )*( q(i+1,j+1) - q(i+1,j-1) )
     &                + p(i-1, j )*( q(i-1,j+1) - q(i-1,j-1) ) )
     &                - ah4fac*d6p
        enddo
*       Eastern boundary (i=nxpa) (periodic)
        dqdt(nxpa,j) = dqdt(1,j)
      enddo
!$OMP END DO NOWAIT

!$OMP END PARALLEL

*     Northern boundary Jacobian term
*     -------------------------------
*     Cyclic advection at Western boundary.
      aj5smn = - 0.5d0*q( 1 , nypa )*( p( 2 ,nypa-1)-p(nxpa-1,nypa-1) )
      aj9smn = - 0.5d0*q( 1 ,nypa-1)*( p( 2 ,nypa-1)-p(nxpa-1,nypa-1) )
*     Internal points
      do i=2,nxpa-1
        aj5smn = aj5smn - q(i, nypa )*( p(i+1,nypa-1)-p(i-1,nypa-1) )
        aj9smn = aj9smn - q(i,nypa-1)*( p(i+1,nypa-1)-p(i-1,nypa-1) )
      enddo
*     Cyclic advection at Eastern boundary.
      aj5smn = aj5smn
     &        - 0.5d0*q(nxpa, nypa )*( p( 2 ,nypa-1)-p(nxpa-1,nypa-1) )
      aj9smn = aj9smn
     &        - 0.5d0*q(nxpa,nypa-1)*( p( 2 ,nypa-1)-p(nxpa-1,nypa-1) )
      ajinat = fnot*adfaca*( aj5smn + 2.0d0*aj9smn )
*     Convert boundary constraint sum to integral
      ajinat = dxa*dya*ajinat
*
*     Compute fifth derivative contributions at zonal
*     boundaries to the momentum constraint equations
*     -----------------------------------------------
      ah5sms = 0.5d0*( d4p(1,  2 ) - d4p(1,   1  ) )
      ah5smn = 0.5d0*( d4p(1,nypa) - d4p(1,nypa-1) )
      do i=2,nxpa-1
        ah5sms = ah5sms + ( d4p(i,  2 ) - d4p(i,   1  ) )
        ah5smn = ah5smn + ( d4p(i,nypa) - d4p(i,nypa-1) )
      enddo
      ah5sms = ah5sms + 0.5d0*( d4p(nxpa,  2 ) - d4p(nxpa,   1  ) )
      ah5smn = ah5smn + 0.5d0*( d4p(nxpa,nypa) - d4p(nxpa,nypa-1) )
      ahp5is = ah4atk*ah5sms
      ahp5in = ah4atk*ah5smn

      END SUBROUTINE atadif

#endif /* not ocean_only */
c
c********************************************************************
c    
#if (!defined ocean_only) && (defined lu_mode)

      SUBROUTINE atadslu (dqdt, q, p, ur, vr, axx, ayy, axy)

*     Compute the total variation of PV in LU formulation, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxta,nxpa,nyta,nypa,beta
      USE atconst, ONLY : dxa,rdxaf0,hdxam1,dxam2,bccoat 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dqdt(nxpa,nypa)
      double precision, INTENT(IN) :: q(nxpa,nypa),p(nxpa,nypa),
     &                                ur(nxpa,nyta),vr(nxta,nypa),
     &                                axx(nxpa,nyta),ayy(nxta,nypa),
     &                                axy(nxpa,nypa)
*
*     dqdt is total varaition of PV (s^-1)
*     q and p are PV (s^-1) and pressure (m^2 s^-2)
*     ur and vr are small-scale random velocities (m s^-1)
*     axx, ayy and axy are small-scale diffusion tensors (m^2 s^-1)       

*     Local variables          
      integer i,j
      double precision, DIMENSION(nxpa,nyta) :: ul,us,dyul 
      double precision, DIMENSION(nxta,nypa) :: vl,vs,dxvl 
      double precision bcfac,adfac,difac,sofac,sifac,adfxp,adfyp,
     &                 adfxm,adfym,advs,difxp,difyp,difxm,difym,difa,
     &                 sofxp,sofyp,sofxm,sofym,srcs,sifxp,sifyp,sifxm,
     &                 sifym,srca,sbfac
*
*     ul and vl are large-scale velocities (m s^-1)
*     us and vs are small-scale velocities (m s^-1), which are random 
*     velocities plus Ito-Stokes drift
*     dyul is difference of u in y-axis      
*     dxvl is difference of v in x-axis      
*     adfxm and adfym is in-flux vector of small-scale advection advs
*     adfxp and adfyp is out-flux vector of small-scale advection advs
*     difxm and difym is in-flux vector of small-scale diffusion difa
*     difxp and difyp is out-flux vector of small-scale diffusion difa
*     sofxm and sofym is in-flux vector of small-scale sources srcs
*     sofxp and sofyp is out-flux vector of small-scale sources srcs
*     sifxm and sifym is in-flux vector of small-scale sink srca
*     sifxp and sifyp is out-flux vector of small-scale sink srca

*     Constant factors      
      bcfac = (1.0d0 - 0.5d0*bccoat)/(1.0d0 + 0.5d0*bccoat)
      adfac = hdxam1/6.0d0
      difac = dxam2/8.0d0
      sofac = dxam2/2.0d0
      sifac = dxam2/(4.0d0*dxa)
      sbfac = beta*hdxam1

*     Boundary values are not account      
      dqdt = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym,advs,difxp,
!$OMP&                  difxm,difyp,difym,difa,sofxp,sofxm,sofyp,
!$OMP&                  sofym,srcs,sifxp,sifxm,sifyp,sifym,srca)
!$OMP&         SHARED  (ul,us,ur,vl,vs,vr,p,q,axx,axy,ayy,dyul,
!$OMP&                  dxvl,rdxaf0,hdxam1,bcfac,adfac,difac,
!$OMP&                  sofac,sifac,sbfac,dqdt)      

*     Compute large- and small-scale velocities
!$OMP DO SCHEDULE (STATIC)      
*     U-velocity
      do j=1,nyta
*       Western boundary
        ul(1,j) = - rdxaf0*( p(1,j+1) - p(1,j) ) 
        us(1,j) = ur(1,j) - hdxam1*( axy(1,j+1) - axy(1,j) +
     &                        0.5d0*(axx(2,j) - axx(nxta,j)) )
*       Inner points
        do i=2,nxta
          ul(i,j) = - rdxaf0*( p(i,j+1) - p(i,j) ) 
          us(i,j) = ur(i,j) - hdxam1*( axy(i,j+1) - axy(i,j) +
     &                        0.5d0*(axx(i+1,j) - axx(i-1,j)) )   
        enddo
*       Eastern boundary
        ul(nxpa,j) = ul(1,j)
        us(nxpa,j) = us(1,j)
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
*     V-velocity
      do i=1,nxta
*       Southern boundary
        vl(i,1) = 0.0d0        
        vs(i,1) = 0.0d0 
*       Inner points
        do j=2,nyta
          vl(i,j) = rdxaf0*( p(i+1,j) - p(i,j) )
          vs(i,j) = vr(i,j) - hdxam1*( axy(i+1,j) - axy(i,j) +
     &                        0.5d0*(ayy(i,j+1) - ayy(i,j-1)) )   
        enddo
*       Northern boundary
        vl(i,nypa) = 0.0d0
        vs(i,nypa) = 0.0d0 
      enddo
!$OMP END DO

*     Compute diff. of ul in y-axis
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpa
*       Western boundary
        dyul(i,1) = ul(i,2) - bcfac*ul(i,1)
*       Inner points
        do j=2,nyta-1
          dyul(i,j) = ul(i,j+1) - ul(i,j-1)
        enddo
*       Eastern boundary
        dyul(i,nyta) = bcfac*ul(i,nyta) - ul(i,nyta-1)
      enddo
!$OMP END DO

*     Compute diff. of v in x-axis
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypa
*       Southern boundary
        dxvl(1,j) = vl(2,j) - vl(nxta,j) !!TODO: make sure cycle
*       Inner points
        do i=2,nxta-1
          dxvl(i,j) = vl(i+1,j) - vl(i-1,j)
        enddo
*       Northern boundary
        dxvl(nxta,j) = vl(1,j) - vl(nxta-1,j)
      enddo
!$OMP END DO

*     Compute variation of PV for inner points 
!$OMP DO SCHEDULE (STATIC)      
      do j=2,nyta
        do i=2,nxta
*         **************   Small-scale advection   *************        
*         Outflux in x-axis
          adfxp = (us(i,j) + us(i+1,j) + us(i,j-1) + us(i+1,j-1)) * 
     &            (q(i,j) + q(i+1,j)) + 0.5d0*( (us(i,j) + us(i+1,j))*
     &            (q(i,j) + q(i+1,j+1)) + (us(i,j-1) + us(i+1,j-1))*
     &            (q(i,j) + q(i+1,j-1)) )
*         Influx in x-axis
          adfxm = (us(i-1,j) + us(i-1,j-1) + us(i,j-1) + us(i,j)) * 
     &            (q(i,j) + q(i-1,j)) + 0.5d0*( (us(i,j) + us(i-1,j))*
     &            (q(i,j) + q(i-1,j+1)) + (us(i-1,j-1) + us(i,j-1))*
     &            (q(i,j) + q(i-1,j-1)) )
*         Outflux in y-axis
          adfyp = (vs(i-1,j+1) + vs(i-1,j) + vs(i,j) + vs(i,j+1)) * 
     &            (q(i,j) + q(i,j+1)) + 0.5d0*( (vs(i,j) + vs(i,j+1))*
     &            (q(i,j) + q(i+1,j+1)) + (vs(i-1,j+1) + vs(i-1,j))*
     &            (q(i,j) + q(i-1,j+1)) )
*         Influx in y-axis
          adfym = (vs(i-1,j) + vs(i-1,j-1) + vs(i,j-1) + vs(i,j)) * 
     &            (q(i,j) + q(i,j-1)) + 0.5d0*( (vs(i,j) + vs(i,j-1))*
     &            (q(i,j) + q(i+1,j-1)) + (vs(i-1,j) + vs(i-1,j-1))*
     &            (q(i,j) + q(i-1,j-1)) )
*         Net flux
          advs = - adfac*( adfxp - adfxm + adfyp - adfym )
*         **************   Small-scale diffusion   *************   
*         Outflux in x-axis
          difxp = (axx(i,j) + axx(i,j-1) + axx(i+1,j-1) + axx(i+1,j))*
     &            (q(i+1,j) - q(i,j)) + 0.5d0*(axy(i,j) + axy(i+1,j))*
     &            (q(i,j+1) - q(i,j-1) + q(i+1,j+1) - q(i+1,j-1))
*         Influx in x-axis
          difxm = (axx(i-1,j) + axx(i-1,j-1) + axx(i,j-1) + axx(i,j))*
     &            (q(i,j) - q(i-1,j)) + 0.5d0*(axy(i-1,j) + axy(i,j))*
     &            (q(i-1,j+1) - q(i-1,j-1) + q(i,j+1) - q(i,j-1))
*         Outflux in y-axis
          difyp = (ayy(i,j+1) + ayy(i-1,j) + ayy(i,j) + ayy(i,j+1))*
     &            (q(i,j+1) - q(i,j)) + 0.5d0*(axy(i,j+1) + axy(i,j))*
     &            (q(i+1,j+1) - q(i-1,j+1) + q(i+1,j) - q(i-1,j))
*         Influx in y-axis
          difym = (ayy(i-1,j) + ayy(i-1,j-1) + ayy(i,j-1) + ayy(i,j))*
     &            (q(i,j) - q(i,j-1)) + 0.5d0*(axy(i,j-1) + axy(i,j))*
     &            (q(i+1,j) - q(i-1,j) + q(i+1,j-1) - q(i-1,j-1))
*         Net flux
          difa = difac*( difxp - difxm + difyp - difym )
*         **************     Small-scale sources     *************   
*         Outflux in x-axis
          sofxp = 0.25d0*( ul(i,j) + ul(i,j-1) + 
     &                     ul(i+1,j-1) + ul(i+1,j) )
     &                  *( us(i,j) - us(i,j-1) + 
     &                     us(i+1,j) - us(i+1,j-1) )
     &            - vl(i,j)*( us(i+1,j) - us(i,j) + 
     &                        us(i+1,j-1) - us(i,j-1) )
*         Influx in x-axis
          sofxm = 0.25d0*( ul(i-1,j) + ul(i-1,j-1) + 
     &                     ul(i,j-1) + ul(i,j) )
     &                  *( us(i-1,j) - us(i-1,j-1) + 
     &                     us(i,j) - us(i,j-1) ) 
     &            - vl(i-1,j)*( us(i,j-1) - us(i-1,j-1) + 
     &                          us(i,j) - us(i-1,j) )
*         Outflux in y-axis
          sofyp = - 0.25d0*( vl(i-1,j+1) + vl(i-1,j) + 
     &                       vl(i,j) + vl(i,j+1) )
     &                    *( vs(i,j+1) - vs(i-1,j+1) + 
     &                       vs(i,j) - vs(i-1,j) )
     &            + ul(i,j)*( vs(i-1,j+1) - vs(i-1,j) + 
     &                        vs(i,j+1) - vs(i,j) )
*         Influx in y-axis
          sofym = - 0.25d0*( vl(i-1,j) + vl(i-1,j-1) + 
     &                       vl(i,j) + vl(i,j-1) )
     &                    *( vs(i,j) - vs(i-1,j) + 
     &                       vs(i,j-1) - vs(i-1,j-1) )
     &            + ul(i,j-1)*( vs(i-1,j) - vs(i-1,j-1) + 
     &                          vs(i,j) - vs(i,j-1) )
*         Net flux 
          srcs = sofac*( sofxp - sofxm + sofyp - sofym )
*         **************     Small-scale sinks     *************   
*         Outflux in x-axis
          sifxp = 0.50d0*( axx(i+1,j) - axx(i,j) + 
     &                     axx(i+1,j-1) + axx(i,j-1) )*dxvl(i,j)
     &          +        ( axy(i+1,j) - axy(i,j) )
     &                  *( vl(i,j+1) - vl(i,j-1) ) 
     &          - 0.50d0*( axx(i,j) - axx(i,j-1) + 
     &                     axx(i+1,j) - axx(i+1,j-1) )
     &                  *( ul(i+1,j) - ul(i,j) + 
     &                     ul(i+1,j-1) - ul(i,j-1) )
     &          - 0.25d0*( axy(i,j+1) - axy(i,j-1) + 
     &                     axy(i+1,j+1) - axy(i+1,j-1) )
     &                  *( ul(i,j) - ul(i,j-1) + 
     &                     ul(i+1,j) - ul(i+1,j-1) )
*         Influx in x-axis
          sifxm = 0.50d0*( axx(i,j) - axx(i-1,j) + 
     &                     axx(i,j-1) + axx(i-1,j-1) )*dxvl(i-1,j)
     &          +        ( axy(i,j) - axy(i-1,j) )
     &                  *( vl(i-1,j+1) - vl(i-1,j-1) ) 
     &          - 0.50d0*( axx(i-1,j) - axx(i-1,j-1) + 
     &                     axx(i,j) - axx(i,j-1) )
     &                  *( ul(i,j) - ul(i-1,j) + 
     &                     ul(i,j-1) - ul(i-1,j-1) )
     &          - 0.25d0*( axy(i-1,j+1) - axy(i-1,j-1) + 
     &                     axy(i,j+1) - axy(i,j-1) )
     &                  *( ul(i-1,j) - ul(i-1,j-1) + 
     &                     ul(i,j) - ul(i,j-1) )
*         Outflux in y-axis
          sifyp = 0.25d0*( axy(i+1,j+1) - axy(i-1,j+1) + 
     &                     axy(i+1,j) - axy(i-1,j) )
     &                  *( vl(i,j+1) - vl(i-1,j+1) + 
     &                     vl(i,j) - vl(i-1,j) )
     &          + 0.50d0*( ayy(i,j+1) - ayy(i-1,j+1) + 
     &                     ayy(i,j) - ayy(i-1,j) )
     &                  *( vl(i-1,j+1) - vl(i-1,j) + 
     &                     vl(i,j+1) - vl(i,j) )
     &          -        ( axy(i,j+1) - axy(i,j) )
     &                  *( ul(i+1,j) - ul(i-1,j) ) 
     &          - 0.50d0*( ayy(i-1,j+1) - ayy(i-1,j) + 
     &                     ayy(i,j+1) - ayy(i,j) )*dyul(i,j)
*         Influx in y-axis
          sifym = 0.25d0*( axy(i+1,j) - axy(i-1,j) + 
     &                     axy(i+1,j-1) - axy(i-1,j-1) )
     &                  *( vl(i,j) - vl(i-1,j) + 
     &                     vl(i,j-1) - vl(i-1,j-1) )
     &          + 0.50d0*( ayy(i,j) - ayy(i-1,j) + 
     &                     ayy(i,j-1) - ayy(i-1,j-1) )
     &                  *( vl(i-1,j) - vl(i-1,j-1) + 
     &                     vl(i,j) - vl(i,j-1) )
     &          -        ( axy(i,j) - axy(i,j-1) )
     &                  *( ul(i+1,j-1) - ul(i-1,j-1) ) 
     &          - 0.50d0*( ayy(i-1,j) - ayy(i-1,j-1) + 
     &                     ayy(i,j) - ayy(i,j-1) )*dyul(i,j-1)
*         Net flux
          srca = sifac*( sifxp - sifxm + sifyp - sifym )
*         Total variation of PV
          dqdt(i,j) = advs + difa + srcs + srca 
     &              - sbfac*( axy(i+1,j) - axy(i-1,j) + 
     &                0.5d0*( ayy(i-1,j+1) - ayy(i-1,j-1) +
     &                        ayy(i,j+1) - ayy(i,j-1) ) )     
        enddo
      enddo
!$OMP END DO
!$OMP END PARALLEL

      END SUBROUTINE atadslu

#endif /* (not ocean_only) && (lu_mode) */
c
c********************************************************************
c    
#if (!defined ocean_only) && (defined use_edvel)

      SUBROUTINE ataded (dqdt, q, u, v)

*     Add a subgrid advection by eddy velocities, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxta,nxpa,nyta,nypa
      USE atconst, ONLY : hdxam1 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dqdt(nxpa,nypa)
      double precision, INTENT(IN) :: q(nxpa,nypa),
     &                                u(nxpa,nyta),v(nxta,nypa)

*     Local variables      
      integer i,j
      double precision adfac,adfxp,adfyp,adfxm,adfym

      adfac = hdxam1/6.0d0
!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym)
!$OMP&         SHARED  (dqdt,u,v,q,adfac,hdxam1)      

!$OMP DO SCHEDULE (STATIC)      
      do j=2,nyta
        do i=2,nxta
*         Outflux in x-axis
          adfxp = (u(i,j) + u(i+1,j) + u(i,j-1) + u(i+1,j-1)) * 
     &            (q(i,j) + q(i+1,j)) + 0.5d0*( (u(i,j) + u(i+1,j))*
     &            (q(i,j) + q(i+1,j+1)) + (u(i,j-1) + u(i+1,j-1))*
     &            (q(i,j) + q(i+1,j-1)) )
*         Influx in x-axis
          adfxm = (u(i-1,j) + u(i-1,j-1) + u(i,j-1) + u(i,j)) * 
     &            (q(i,j) + q(i-1,j)) + 0.5d0*( (u(i,j) + u(i-1,j))*
     &            (q(i,j) + q(i-1,j+1)) + (u(i-1,j-1) + u(i,j-1))*
     &            (q(i,j) + q(i-1,j-1)) )
*         Outflux in y-axis
          adfyp = (v(i-1,j+1) + v(i-1,j) + v(i,j) + v(i,j+1)) * 
     &            (q(i,j) + q(i,j+1)) + 0.5d0*( (v(i,j) + v(i,j+1))*
     &            (q(i,j) + q(i+1,j+1)) + (v(i-1,j+1) + v(i-1,j))*
     &            (q(i,j) + q(i-1,j+1)) )
*         Influx in y-axis
          adfym = (v(i-1,j) + v(i-1,j-1) + v(i,j-1) + v(i,j)) * 
     &            (q(i,j) + q(i,j-1)) + 0.5d0*( (v(i,j) + v(i,j-1))*
     &            (q(i,j) + q(i+1,j-1)) + (v(i-1,j) + v(i-1,j-1))*
     &            (q(i,j) + q(i-1,j-1)) )
*         Net flux
          dqdt(i,j) = -adfac*( adfxp - adfxm + adfyp - adfym )
        enddo
      enddo  
!$OMP END DO
!$OMP END PARALLEL      

      END SUBROUTINE ataded

#endif /* (not ocean_only) && (use_edvel) */
c
c***********************************************************************
c
      END MODULE qgasubs
c
c***********************************************************************
