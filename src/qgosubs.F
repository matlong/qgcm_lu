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
      MODULE qgosubs

*     Contains subroutines qgostep and ocadif which step
*     the oceanic vorticity evolution equation (7.14).

*     Modules

      IMPLICIT NONE

      PRIVATE

#ifndef atmos_only
      PUBLIC  :: qgostep
      PRIVATE :: ocadif
#  ifdef lu_mode      
      PRIVATE :: ocadslu
#  endif
#  ifdef use_edvel
      PRIVATE :: ocaded
#  endif      

      CONTAINS
#endif /* not atmos_only */

c***********************************************************************
c
#ifndef atmos_only

      SUBROUTINE qgostep
*
*     Steps the evolution equation (7.14) for
*     the oceanic vorticity. Leapfrog timestep.

*     Modules
      USE parameters
      USE occonst
      USE ocstate
#  ifdef cyclic_ocean
      USE ochomog
#  endif
#  ifdef lu_mode      
      USE ocluvar
#  endif
#  ifdef use_edvel
      USE ocedvl
#  endif      

      IMPLICIT NONE

*     Subroutine arguments
*
*     Local variables
      integer i,j,k
      double precision adfaco,bcfaco,fohfac(nlo),bdrfac,
     &                 del2p(nxpo,nypo),dqdt(nxpo,nypo,nlo),
     &                 qdot(nlo),qold,dqdtlu(nxpo,nypo,nlo),
     &                 dqdted(nxpo,nypo,nlo),
     &                 dqmdtlu(nxpo,nypo,nlo)
#  ifdef cyclic_ocean
      double precision bdsums,bdsumn
#  endif

      adfaco = 1.0d0/(12.0d0*dxo*dyo*fnot)
*     Version with nondimensional bccooc
      bcfaco = bccooc*dxom2/( 0.5d0*bccooc + 1.0d0 )
      do k=1,nlo
        fohfac(k) = fnot/hoc(k)
      enddo
      bdrfac = 0.5d0*sign(1.0d0,fnot)*delek/hoc(nlo)

c     OCEANIC q STEP

      do k=1,nlo

*       Compute Del-sqd(p) at previous time level for dissipation
*       ---------------------------------------------------------
!$OMP   PARALLEL DEFAULT (NONE)
!$OMP&           PRIVATE (i,j)
!$OMP&           SHARED  (k,del2p,bcfaco,pom,dxom2)

*       N & S boundaries (including corners) - mixed BCs
*       No x-derivative term because p is constant along boundary
!$OMP   DO SCHEDULE (STATIC)
        do i=1,nxpo
          del2p(i,  1 ) = bcfaco*( pom(i,   2  ,k) - pom(i,  1 ,k) )
          del2p(i,nypo) = bcfaco*( pom(i,nypo-1,k) - pom(i,nypo,k) )
        enddo
!$OMP   END DO NOWAIT

!$OMP   DO SCHEDULE (STATIC)
        do j=2,nypo-1
*         Western boundary (i=1)
#  ifdef cyclic_ocean
*         Zonally cyclic ocean
          del2p(1,j) = (  pom(1,j-1,k) + pom(nxpo-1,j,k) + pom(2,j,k)
     &                  + pom(1,j+1,k) - 4.0d0*pom(1,j,k) )*dxom2
#  else
*         Finite box ocean - mixed BCs
          del2p( 1 ,j) = bcfaco*( pom(  2  ,j,k) - pom( 1 ,j,k) )
#  endif
*         Inner points (standard case)
          do i=2,nxpo-1
            del2p(i,j) = (  pom(i,j-1,k) + pom(i-1,j,k) + pom(i+1,j,k)
     &                    + pom(i,j+1,k) - 4.0d0*pom(i,j,k) )*dxom2
          enddo
*         Eastern boundary (i=nxpo)
#  ifdef cyclic_ocean
*         Zonally cyclic ocean
          del2p(nxpo,j) = del2p(1,j)
#  else
*         Finite box ocean - mixed BCs
          del2p(nxpo,j) = bcfaco*( pom(nxpo-1,j,k) - pom(nxpo,j,k) )
#  endif
        enddo
!$OMP   END DO NOWAIT

!$OMP   END PARALLEL

*       Further differentiate contents of del2p to get Del-6th(lagged p).
*       Initialise dqdt in each layer with (Jacobian)
*       advective term plus Del-4th and Del-6th frictional terms.
*       Include (1/fnot) factor in multiplicative factor adfaco
#  ifdef cyclic_ocean
*       Zonally cyclic ocean
        call ocadif (dqdt(1,1,k), del2p, ah2oc(k), ah4oc(k), bcfaco,
     &               po(1,1,k), qo(1,1,k), adfaco, ajisoc(k), ajinoc(k),
     &               ap3soc(k), ap5soc(k), ap3noc(k), ap5noc(k))

#  else
*       Finite box ocean
        call ocadif (dqdt(1,1,k), del2p, ah2oc(k), ah4oc(k), bcfaco,
     &               po(1,1,k), qo(1,1,k), adfaco)
#  endif

#  ifdef lu_mode
        call ocadslu (dqmdtlu(1,1,k), qom(1,1,k), pom(1,1,k), 
     &                uro(1,1,k), vro(1,1,k), 
     &                aoxx(1,1,k), aoyy(1,1,k), aoxy(1,1,k))
        call ocadslu (dqdtlu(1,1,k), qo(1,1,k), po(1,1,k), 
     &                uro(1,1,k), vro(1,1,k), 
     &                aoxx(1,1,k), aoyy(1,1,k), aoxy(1,1,k))
#  else
        dqdtlu = 0.0d0
        dqmdtlu = 0.0d0
#  endif /* lu_mode */

#  ifdef use_edvel        
        call ocaded (dqdted(1,1,k), qo(1,1,k), po(1,1,k),
     &               uoed(1,1,k), voed(1,1,k))
#  else
        dqdted = 0.0d0
#  endif /* use_edvel */

      enddo
*
#  ifdef cyclic_ocean
*     Compute bottom drag first derivative contributions at
*     zonal boundaries to the momentum constraint equations
*     -----------------------------------------------------
*     Zonally cyclic ocean
      bdsums = 0.0d0
      bdsumn = 0.0d0
      do i=1,nxpo-1
        bdsums = bdsums + ( pom(i,  2 ,nlo) - pom(i,   1  ,nlo) )
        bdsumn = bdsumn + ( pom(i,nypo,nlo) - pom(i,nypo-1,nlo) )
      enddo
      bdrins = 0.5d0*sign(1.0d0,fnot)*delek*bdsums
      bdrinn = 0.5d0*sign(1.0d0,fnot)*delek*bdsumn
#  endif

*     Specify forcing, diabatics and bottom friction; then timestep
*     -------------------------------------------------------------
*     These are the layer-specific terms near the top and bottom
*     At this point the array del2p contains Del-sqd(lagged p)
*     for the bottom layer k = nlo

!$OMP PARALLEL DO DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,k,qdot,qold)
!$OMP&         SHARED  (dqdt,fohfac,wekpo,entoc,
!$OMP&                  bdrfac,del2p,qo,qom,tdto,
!$OMP&                  dto,dqdtlu,dqdted,dqmdtlu)
!$OMP&         SCHEDULE (STATIC)
      do j=2,nypo-1
        do i=1,nxpo
          qdot( 1 ) = dqdt(i,j,1)
     &               + fohfac(1)*( wekpo(i,j) - entoc(i,j) )
          qdot( 2 ) = dqdt(i,j,2) + fohfac(2)*entoc(i,j)
          do k=3,nlo
            qdot(k) = dqdt(i,j,k)
          enddo
*         Add linear bottom drag
          qdot(nlo) = qdot(nlo) - bdrfac*del2p(i,j)
*         Step the values of qo except at zonal boundaries
*         The boundary values of qo will be updated later by ocqbdy,
*         including meridional boundaries for the finite box case
          do k=1,nlo
            qold = qo(i,j,k)
            qo(i,j,k) = qom(i,j,k) + tdto*qdot(k)
     &                             + dto*dqmdtlu(i,j,k)
     &                             + dto*dqdtlu(i,j,k)       
     &                             + tdto*dqdted(i,j,k)       
            qom(i,j,k) = qold
          enddo
        enddo
      enddo
!$OMP END PARALLEL DO

*     Copy zonal boundary values, which have not been updated
*     Boundary points are all updated later by ocqbdy
      do k=1,nlo
        do i=1,nxpo
          qom(i,  1 ,k) = qo(i,  1 ,k)
          qom(i,nypo,k) = qo(i,nypo,k)
        enddo
      enddo

      END SUBROUTINE qgostep

#endif /* not atmos_only */
c
c***********************************************************************
c
#ifndef atmos_only

#  ifdef cyclic_ocean
*     Zonally cyclic ocean
      SUBROUTINE ocadif (dqdt, d2p, ah2ock, ah4ock, bcfaco,
     &                   p, q, adfaco, ajisoc, ajinoc,
     &                   ahp3is, ahp5is, ahp3in, ahp5in)
#  else
*     Finite box ocean
      SUBROUTINE ocadif (dqdt, d2p, ah2ock, ah4ock, bcfaco,
     &                   p, q, adfaco)
#  endif
*
*     Computes advective + diffusive contributions to evolution
*     equation for q within an oceanic layer. Computes friction
*     terms Del-4th(p) and Del-6th(p). Initialises array dqdt
*     with adfaco times Jacobian advection term J(q,p) plus
*     ah2fac times Del-4th(p) minus ah4fac times Del-6th(p).
*     Uses Arakawa energy and enstrophy conserving 9-point Jacobian
*     Also computes the near-boundary contributions to the momentum
*     constraint equations: Ah2*Int ( d3p/dy3 ) dx,
*     Ah4*Int ( d5p/dy5 ) dx and Int ( Jacobian )
*     Array "d2p" contains on entry Del-sqd(time-lagged p),
*     and is unchanged on exit.

*     Modules
      USE parameters
      USE occonst

      IMPLICIT NONE

*     Subroutine arguments
      double precision, INTENT(IN) :: d2p(nxpo,nypo),ah2ock,ah4ock,
     &                                bcfaco,p(nxpo,nypo),q(nxpo,nypo),
     &                                adfaco
      double precision, INTENT(OUT) :: dqdt(nxpo,nypo)
#  ifdef cyclic_ocean
      double precision, INTENT(OUT) :: ajisoc,ajinoc,
     &                                 ahp3is,ahp5is,ahp3in,ahp5in
#  endif
*
*     Local variables
      integer i,j
      double precision ah2fac,ah4fac,d4p(nxpo,nypo),d6p,diffus
#  ifdef cyclic_ocean
      double precision ah3sms,ah3smn,ah5sms,ah5smn,
     &                 aj5sms,aj5smn,aj9sms,aj9smn
#  endif

      ah2fac = ah2ock/fnot
      ah4fac = ah4ock/fnot

#  ifdef cyclic_ocean
*     Southern boundary Jacobian term
*     -------------------------------
*     Only needed in the cyclic ocean case
*     Cyclic advection at Western boundary.
      aj5sms = 0.5d0*q(  1 , 1 )*( p( 2 , 2 )-p(nxpo-1, 2 ) )
      aj9sms = 0.5d0*q(  1 , 2 )*( p( 2 , 2 )-p(nxpo-1, 2 ) )
*     Internal points
      do i=2,nxpo-1
        aj5sms = aj5sms + q(i, 1 )*( p(i+1, 2 )-p(i-1, 2 ) )
        aj9sms = aj9sms + q(i, 2 )*( p(i+1, 2 )-p(i-1, 2 ) )
      enddo
*     Cyclic advection at Eastern boundary.
      aj5sms = aj5sms + 0.5d0*q(nxpo, 1 )*( p( 2 , 2 )-p(nxpo-1, 2 ) )
      aj9sms = aj9sms + 0.5d0*q(nxpo, 2 )*( p( 2 , 2 )-p(nxpo-1, 2 ) )
      ajisoc = fnot*adfaco*( aj5sms + 2.0d0*aj9sms )
*     Convert boundary constraint sum to integral
      ajisoc = dxo*dyo*ajisoc
#  endif
c
c     COMPUTE HORIZONTAL FRICTION - OCEAN
c
!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,d6p,diffus)
!$OMP&         SHARED  (d4p,bcfaco,d2p,dxom2,dqdt,
!$OMP&                  ah2fac,ah4fac,adfaco,q,p)

*     Compute Del-4th(p)
*     ------------------
*     Apply N & S boundary condition: mixed
!$OMP DO SCHEDULE (STATIC)
      do i=1,nxpo
        d4p(i,  1 ) = bcfaco*( d2p(i,   2  ) - d2p(i,  1 ) )
        d4p(i,nypo) = bcfaco*( d2p(i,nypo-1) - d2p(i,nypo) )
      enddo
!$OMP END DO NOWAIT

*     Compute Del-4th(p) except at N & S boundaries
!$OMP DO SCHEDULE (STATIC)
      do j=2,nypo-1
*       Western boundary
#  ifdef cyclic_ocean
*       Zonally cyclic ocean
        d4p( 1 ,j) = (  d2p(1,j-1) + d2p(nxpo-1,j) + d2p(2,j)
     &                + d2p(1,j+1) - 4.0d0*d2p(1,j) )*dxom2
#  else
*       Finite box ocean - mixed BCs
        d4p( 1 ,j) = bcfaco*( d2p(  2  ,j) - d2p( 1 ,j) )
#  endif
*       Inner points (standard case)
        do i=2,nxpo-1
          d4p(i,j) = dxom2*(  d2p(i,j-1) + d2p(i-1,j) + d2p(i+1,j)
     &                      + d2p(i,j+1) - 4.0d0*d2p(i,j) )
        enddo
*       Eastern boundary
#  ifdef cyclic_ocean
*       Zonally cyclic ocean
        d4p(nxpo,j) = d4p(1,j)
#  else
*       Finite box ocean - mixed BCs
        d4p(nxpo,j) = bcfaco*( d2p(nxpo-1,j) - d2p(nxpo,j) )
#  endif
      enddo
!$OMP END DO

*     Compute advective (Jacobian) and diffusive contributions to dq/dt
*     -----------------------------------------------------------------
*     Compute Del-6th(p) except at N & S boundaries
*     No need to apply N & S boundary conditions
*     Initialise dq/dt with Jacobian advection and diffusive terms
!$OMP DO SCHEDULE (STATIC)
      do j=2,nypo-1
*       Western boundary (i=1)
#  ifdef cyclic_ocean
*       Zonally cyclic ocean
        d6p = dxom2*(  d4p(1,j-1) + d4p(nxpo-1,j) + d4p(2,j)
     &               + d4p(1,j+1) - 4.0d0*d4p(1,j) )
        diffus = ah2fac*d4p(1,j) - ah4fac*d6p
        dqdt(1,j) = adfaco*
     &             ( (q( 2 , j )-q(nxpo-1, j ))*(p( 1 ,j+1)-p(  1 ,j-1))
     &              +(q( 1 ,j-1)-q(  1 ,j+1))*(p( 2 , j )-p(nxpo-1, j ))
     &              + q(   2  , j )*( p(   2  ,j+1) - p(   2  ,j-1) )
     &              - q(nxpo-1, j )*( p(nxpo-1,j+1) - p(nxpo-1,j-1) )
     &              - q(   1  ,j+1)*( p(   2  ,j+1) - p(nxpo-1,j+1) )
     &              + q(   1  ,j-1)*( p(   2  ,j-1) - p(nxpo-1,j-1) )
     &              + p(   1  ,j+1)*( q(   2  ,j+1) - q(nxpo-1,j+1) )
     &              - p(   1  ,j-1)*( q(   2  ,j-1) - q(nxpo-1,j-1) )
     &              - p(   2  , j )*( q(   2  ,j+1) - q(   2  ,j-1) )
     &              + p(nxpo-1, j )*( q(nxpo-1,j+1) - q(nxpo-1,j-1) ) )
     &              + diffus
#  else
*       Finite box ocean
        dqdt(1,j) = 0.0d0
#  endif
*       Inner points (standard case)
        do i=2,nxpo-1
          d6p = dxom2*(  d4p(i,j-1) + d4p(i-1,j) + d4p(i+1,j)
     &                 + d4p(i,j+1) - 4.0d0*d4p(i,j) )
          diffus = ah2fac*d4p(i,j) - ah4fac*d6p
          dqdt(i,j) = adfaco*
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
     &                + diffus
        enddo
*       Eastern boundary (i=nxpo)
#  ifdef cyclic_ocean
*       Zonally cyclic ocean
        dqdt(nxpo,j) = dqdt(1,j)
#  else
*       Finite box ocean
        dqdt(nxpo,j) = 0.0d0
#  endif
      enddo
!$OMP END DO NOWAIT

!$OMP END PARALLEL

#  ifdef cyclic_ocean
*     Northern boundary Jacobian term
*     -------------------------------
*     Only needed in the cyclic ocean case
*     Cyclic advection at Western boundary.
      aj5smn = - 0.5d0*q( 1 , nypo )*( p( 2 ,nypo-1)-p(nxpo-1,nypo-1) )
      aj9smn = - 0.5d0*q( 1 ,nypo-1)*( p( 2 ,nypo-1)-p(nxpo-1,nypo-1) )
*     Internal points
      do i=2,nxpo-1
        aj5smn = aj5smn - q(i, nypo )*( p(i+1,nypo-1)-p(i-1,nypo-1) )
        aj9smn = aj9smn - q(i,nypo-1)*( p(i+1,nypo-1)-p(i-1,nypo-1) )
      enddo
*     Cyclic advection at Eastern boundary.
      aj5smn = aj5smn
     &        - 0.5d0*q(nxpo, nypo )*( p( 2 ,nypo-1)-p(nxpo-1,nypo-1) )
      aj9smn = aj9smn
     &        - 0.5d0*q(nxpo,nypo-1)*( p( 2 ,nypo-1)-p(nxpo-1,nypo-1) )
      ajinoc = fnot*adfaco*( aj5smn + 2.0d0*aj9smn )
*     Convert boundary constraint sum to integral
      ajinoc = dxo*dyo*ajinoc
*
*     Compute third and fifth derivative contributions at
*     zonal boundaries to the momentum constraint equations
*     -----------------------------------------------------
*     Only needed in the cyclic ocean case
      ah3sms = 0.0d0
      ah3smn = 0.0d0
      ah5sms = 0.0d0
      ah5smn = 0.0d0
*     Zonally cyclic ocean
      do i=1,nxpo-1
        ah3sms = ah3sms + ( d2p(i,  2 ) - d2p(i,   1  ) )
        ah3smn = ah3smn + ( d2p(i,nypo) - d2p(i,nypo-1) )
        ah5sms = ah5sms + ( d4p(i,  2 ) - d4p(i,   1  ) )
        ah5smn = ah5smn + ( d4p(i,nypo) - d4p(i,nypo-1) )
      enddo
      ahp3is = ah2ock*ah3sms
      ahp3in = ah2ock*ah3smn
      ahp5is = ah4ock*ah5sms
      ahp5in = ah4ock*ah5smn
#  endif /* cyclic_ocean */

      END SUBROUTINE ocadif

#endif /* not atmos_only */
c
c********************************************************************
c    
#if (!defined atmos_only) && (defined lu_mode)

      SUBROUTINE ocadslu (dqdt, q, p, ur, vr, axx, ayy, axy)

*     Compute the total variation of PV in LU formulation, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo,beta
      USE occonst, ONLY : dxo,rdxof0,hdxom1,dxom2,bccooc 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dqdt(nxpo,nypo)
      double precision, INTENT(IN) :: q(nxpo,nypo),p(nxpo,nypo),
     &                                ur(nxpo,nyto),vr(nxto,nypo),
     &                                axx(nxpo,nyto),ayy(nxto,nypo),
     &                                axy(nxpo,nypo)
*
*     dqdt is total varaition of PV (s^-1)
*     q and p are PV (s^-1) and pressure (m^2 s^-2)
*     ur and vr are small-scale random velocities (m s^-1)
*     axx, ayy and axy are small-scale diffusion tensors (m^2 s^-1)       

*     Local variables          
      integer i,j
      double precision, DIMENSION(nxpo,nyto) :: ul,us,dyul 
      double precision, DIMENSION(nxto,nypo) :: vl,vs,dxvl 
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
      bcfac = (1.0d0 - 0.5d0*bccooc)/(1.0d0 + 0.5d0*bccooc)
      adfac = hdxom1/6.0d0
      difac = dxom2/8.0d0
      sofac = dxom2/2.0d0
      sifac = dxom2/(4.0d0*dxo)
      sbfac = beta*hdxom1

*     Boundary values are not account      
      dqdt = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym,advs,difxp,
!$OMP&                  difxm,difyp,difym,difa,sofxp,sofxm,sofyp,
!$OMP&                  sofym,srcs,sifxp,sifxm,sifyp,sifym,srca)
!$OMP&         SHARED  (ul,us,ur,vl,vs,vr,p,q,axx,axy,ayy,dyul,
!$OMP&                  dxvl,rdxof0,hdxom1,bcfac,adfac,difac,
!$OMP&                  sofac,sifac,sbfac,dqdt)      

*     Compute large- and small-scale velocities
!$OMP DO SCHEDULE (STATIC)      
*     U-velocity
      do j=1,nyto
*       Western boundary
        ul(1,j) = 0.0d0 
        us(1,j) = 0.0d0 
*       Inner points
        do i=2,nxto
          ul(i,j) = - rdxof0*( p(i,j+1) - p(i,j) ) 
          us(i,j) = ur(i,j) - hdxom1*( axy(i,j+1) - axy(i,j) +
     &                        0.5d0*(axx(i+1,j) - axx(i-1,j)) )   
        enddo
*       Eastern boundary
        ul(nxpo,j) = 0.0d0
        us(nxpo,j) = 0.0d0
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
*     V-velocity
      do i=1,nxto
*       Southern boundary
        vl(i,1) = 0.0d0        
        vs(i,1) = 0.0d0 
*       Inner points
        do j=2,nyto
          vl(i,j) = rdxof0*( p(i+1,j) - p(i,j) )
          vs(i,j) = vr(i,j) - hdxom1*( axy(i+1,j) - axy(i,j) +
     &                        0.5d0*(ayy(i,j+1) - ayy(i,j-1)) )   
        enddo
*       Northern boundary
        vl(i,nypo) = 0.0d0
        vs(i,nypo) = 0.0d0 
      enddo
!$OMP END DO

*     Compute diff. of ul in y-axis
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpo
*       Western boundary
        dyul(i,1) = ul(i,2) - bcfac*ul(i,1)
*       Inner points
        do j=2,nyto-1
          dyul(i,j) = ul(i,j+1) - ul(i,j-1)
        enddo
*       Eastern boundary
        dyul(i,nyto) = bcfac*ul(i,nyto) - ul(i,nyto-1)
      enddo
!$OMP END DO

*     Compute diff. of v in x-axis
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypo
*       Southern boundary
        dxvl(1,j) = vl(2,j) - bcfac*vl(1,j)
*       Inner points
        do i=2,nxto-1
          dxvl(i,j) = vl(i+1,j) - vl(i-1,j)
        enddo
*       Northern boundary
        dxvl(nxto,j) = bcfac*vl(nxto,j) - vl(nxto-1,j)
      enddo
!$OMP END DO

*     Compute variation of PV for inner points 
!$OMP DO SCHEDULE (STATIC)      
      do j=2,nyto
        do i=2,nxto
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

      END SUBROUTINE ocadslu

#endif /* (not atmos_only) && (lu_mode) */
c
c********************************************************************
c    
#if (!defined atmos_only) && (defined use_edvel)

      SUBROUTINE ocaded (dqdt, q, p, us, vs)

*     Add a subgrid advection by eddy velocities, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo
      USE occonst, ONLY : hdxom1,rdxof0,dxom2 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dqdt(nxpo,nypo)
      double precision, INTENT(IN) :: q(nxpo,nypo),p(nxpo,nypo),
     &                                us(nxpo,nyto),vs(nxto,nypo)

*     Local variables      
      integer i,j
      double precision adfac,adfxp,adfyp,adfxm,adfym,
     &                 sofac,sofxp,sofyp,sofxm,sofym,
     &                 ul(nxpo,nyto),vl(nxto,nypo)

      adfac = hdxom1/6.0d0
      sofac = dxom2/2.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym,
!$OMP&                      sofxp,sofxm,sofyp,sofym)
!$OMP&         SHARED  (dqdt,us,vs,ul,vl,q,p,adfac,sofac,rdxof0)      

!$OMP DO SCHEDULE (STATIC)      
*     U-velocity
      do j=1,nyto
*       Western boundary
        ul(1,j) = 0.0d0 
*       Inner points
        do i=2,nxto
          ul(i,j) = - rdxof0*( p(i,j+1) - p(i,j) ) 
        enddo
*       Eastern boundary
        ul(nxpo,j) = 0.0d0
      enddo
!$OMP END DO NOWAIT

!$OMP DO SCHEDULE (STATIC)      
*     V-velocity
      do i=1,nxto
*       Southern boundary
        vl(i,1) = 0.0d0        
*       Inner points
        do j=2,nyto
          vl(i,j) = rdxof0*( p(i+1,j) - p(i,j) )
        enddo
*       Northern boundary
        vl(i,nypo) = 0.0d0
      enddo
!$OMP END DO

!$OMP DO SCHEDULE (STATIC)      
      do j=2,nyto
        do i=2,nxto
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
          dqdt(i,j) = -adfac*( adfxp - adfxm + adfyp - adfym )
     &                +sofac*( sofxp - sofxm + sofyp - sofym )
        enddo
      enddo  
!$OMP END DO
!$OMP END PARALLEL      

      END SUBROUTINE ocaded

#endif /* (not atmos_only) && (use_edvel) */
c
c***********************************************************************
c
      END MODULE qgosubs
c
c***********************************************************************
