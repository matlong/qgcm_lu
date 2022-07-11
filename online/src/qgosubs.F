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
      PRIVATE :: ocadslu,ocadiflu
#  endif
#  ifdef use_edvel
      PRIVATE :: ocadsed
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
     &                 qdot(nlo),qold,rdzo
#  ifdef cyclic_ocean
      double precision bdsums,bdsumn
#  endif
!!#  if (defined lu_mode) || (defined use_edvel)
!!      double precision bm(nxpo,nypo),dzu(nxpo,nyto),dzv(nxto,nypo)
!!#  endif      
#  ifdef lu_mode
!!      double precision dzaxx(nxpo,nyto),dzayy(nxto,nypo),
!!     &                 dzaxy(nxpo,nypo),dqzlu(nxpo,nypo,nlo-1),
!!     &                 dqdtlu(nxpo,nypo,nlo)
      double precision dqdtlu(nxpo,nypo,nlo)
#  endif
#  ifdef use_edvel
      double precision dqdted(nxpo,nypo,nlo)!!,dqzed(nxpo,nypo,nlo-1)
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
!!        call ocadslu (dqdtlu(1,1,k), qom(1,1,k), pom(1,1,k), 
!!     &                uro(1,1,k), vro(1,1,k), 
!!     &                aoxx(1,1,k), aoyy(1,1,k), aoxy(1,1,k))
        call ocadiflu (dqdtlu(1,1,k), qom(1,1,k),
     &                 uro(1,1,k), vro(1,1,k), 
     &                 aoxx(1,1,k), aoyy(1,1,k), aoxy(1,1,k))
#  endif /* lu_mode */

#  ifdef use_edvel        
!!        call ocadsed (dqdted(1,1,k), qom(1,1,k), pom(1,1,k),
!!     &                uoed(1,1,k), voed(1,1,k))
        call ocadbed (dqdted(1,1,k), qom(1,1,k),
     &                uoed(1,1,k), voed(1,1,k))
#  endif /* use_edvel */

!!        if (k.lt.nlo) then
!!          rdzo = 2.0d0/(hoc(k) + hoc(k+1))
*         Rescaled buoyancy
!!#  if (defined lu_mode) || (defined use_edvel)                
!!$OMP     PARALLEL DO DEFAULT (NONE)
!!$OMP&                PRIVATE (i,j)
!!$OMP&                SHARED  (k,bm,gpoc,pom)
!!$OMP&                SCHEDULE (STATIC)      
!!          do j=1,nypo
!!            do i=1,nxpo
!!              bm(i,j) = fnot*( pom(i,j,k) - pom(i,j,k+1) )/gpoc(k)
!!            enddo
!!          enddo
!!$OMP     END PARALLEL DO
!!#  endif          

!!#  ifdef lu_mode          
*         Vertical derivatives
!!$OMP     PARALLEL DEFAULT (NONE)
!!$OMP&             PRIVATE (i,j)
!!$OMP&             SHARED  (k,dzu,dzv,dzaxx,dzayy,dzaxy,rdzo,
!!$OMP&                      uro,vro,aoxx,aoyy,aoxy)
!!$OMP     DO SCHEDULE (STATIC)      
!!          do j=1,nyto
!!            do i=1,nxto
!!              dzu(i,j) = rdzo*( uro(i,j,k) - uro(i,j,k+1) ) 
!!              dzv(i,j) = rdzo*( vro(i,j,k) - vro(i,j,k+1) ) 
!!              dzaxx(i,j) = rdzo*( aoxx(i,j,k) - aoxx(i,j,k+1) ) 
!!              dzayy(i,j) = rdzo*( aoyy(i,j,k) - aoyy(i,j,k+1) ) 
!!              dzaxy(i,j) = rdzo*( aoxy(i,j,k) - aoxy(i,j,k+1) ) 
!!            enddo
*           (i = nxpo)
!!            dzu(nxpo,j) = rdzo*( uro(nxpo,j,k) - uro(nxpo,j,k+1) ) 
!!            dzaxx(nxpo,j) = rdzo*( aoxx(nxpo,j,k) - aoxx(nxpo,j,k+1) ) 
!!            dzaxy(nxpo,j) = rdzo*( aoxy(nxpo,j,k) - aoxy(nxpo,j,k+1) ) 
!!          enddo
!!$OMP     END DO NOWAIT
*         (j = nypo)
!!$OMP     DO SCHEDULE (STATIC)      
!!          do i=1,nxto
!!            dzv(i,nypo) = rdzo*( vro(i,nypo,k) - vro(i,nypo,k+1) ) 
!!            dzayy(i,nypo) = rdzo*( aoyy(i,nypo,k) - aoyy(i,nypo,k+1) ) 
!!            dzaxy(i,nypo) = rdzo*( aoxy(i,nypo,k) - aoxy(i,nypo,k+1) ) 
!!          enddo
!!$OMP     END DO
!!$OMP     END PARALLEL
*         Additional advection and diffusion
!!          call ocadblu (dqzlu(1,1,k), bm, dzu, dzv, dzaxx, dzayy, dzaxy)
!!#  endif     

!!#  ifdef use_edvel
*         Vertical derivatives
!!$OMP     PARALLEL DEFAULT (NONE)
!!$OMP&             PRIVATE (i,j)
!!$OMP&             SHARED  (k,dzu,dzv,rdzo,uoed,voed)
!!$OMP     DO SCHEDULE (STATIC)      
!!          do j=1,nyto
!!            do i=1,nxpo
!!              dzu(i,j) = rdzo*( uoed(i,j,k) - uoed(i,j,k+1) ) 
!!            enddo
!!          enddo
!!$OMP     END DO NOWAIT
!!$OMP     DO SCHEDULE (STATIC)      
!!          do i=1,nxto
!!            do j=1,nypo
!!              dzv(i,j) = rdzo*( voed(i,j,k) - voed(i,j,k+1) ) 
!!            enddo
!!          enddo
!!$OMP     END DO
!!$OMP     END PARALLEL
*         Additional advection          
!!          call ocadbed (dqzed(1,1,k), bm, dzu, dzv)
!!#  endif
!!        endif

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
!$OMP&                  bdrfac,del2p,qo,qom,tdto)
#  ifdef lu_mode
!!$OMP&         SHARED  (dqdtlu,dqzlu)
!$OMP&         SHARED  (dqdtlu)
#  endif      
#  ifdef use_edvel
!!$OMP&         SHARED  (dqdted,dqzed)
!$OMP&         SHARED  (dqdted)
#  endif      
!$OMP&         SCHEDULE (STATIC)
      do j=2,nypo-1
        do i=1,nxpo
*         Upper layer        
          qdot(1) = dqdt(i,j,1) + fohfac(1)*( wekpo(i,j) - entoc(i,j) )
#  ifdef lu_mode
          qdot(1) = qdot(1) + dqdtlu(i,j,1) !!+ 0.5d0*dqzlu(i,j,1)
#  endif          
#  ifdef use_edvel
          qdot(1) = qdot(1) + dqdted(i,j,1) !!+ 0.5d0*dqzed(i,j,1)
#  endif          
*         Inner layers          
          do k=2,nlo-1
            qdot(k) = dqdt(i,j,k)
#  ifdef lu_mode
            qdot(k) = qdot(k) + dqdtlu(i,j,k) 
!!     &              + 0.5d0*( dqzlu(i,j,k-1) + dqzlu(i,j,k) )
#  endif          
#  ifdef use_edvel
            qdot(k) = qdot(k) + dqdted(i,j,k) 
!!     &              + 0.5d0*( dqzed(i,j,k-1) + dqzed(i,j,k) )
#  endif          
          enddo
*         Add vertical entrainement in layer 2
          qdot(2) = qdot(2) + fohfac(2)*entoc(i,j)
*         Bottom layer
          qdot(nlo) = dqdt(i,j,nlo) - bdrfac*del2p(i,j)
#  ifdef lu_mode
          qdot(nlo) = qdot(nlo) + dqdtlu(i,j,nlo) 
!!     &              + 0.5d0*dqzlu(i,j,nlo-1)
#  endif          
#  ifdef use_edvel
          qdot(nlo) = qdot(nlo) + dqdted(i,j,nlo) 
!!     &              + 0.5d0*dqzed(i,j,nlo-1)
#  endif 
*         Step the values of qo except at zonal boundaries
*         The boundary values of qo will be updated later by ocqbdy,
*         including meridional boundaries for the finite box case
          do k=1,nlo
            qold = qo(i,j,k)
            qo(i,j,k) = qom(i,j,k) + tdto*qdot(k)
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

      SUBROUTINE ocadiflu (dqdt, q, ur, vr, axx, ayy, axy)

*     Compute the total variation of PV in LU formulation, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo
      USE occonst, ONLY : hdxom1,dxom2

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dqdt(nxpo,nypo)
      double precision, INTENT(IN), DIMENSION(nxpo,nypo) :: q,axy
      double precision, INTENT(IN), DIMENSION(nxpo,nyto) :: ur,axx
      double precision, INTENT(IN), DIMENSION(nxto,nypo) :: vr,ayy

*     Local variables          
      integer i,j
      double precision us(nxpo,nyto),vs(nxto,nypo),adfac,difac,
     &                 adfxp,adfyp,adfxm,adfym,difxp,difyp,difxm,difym

*     Constant factors      
      adfac = hdxom1/6.0d0
      difac = dxom2/8.0d0

*     Boundary values are not account      
      dqdt = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym,difxp,difxm,difyp,difym)
!$OMP&         SHARED  (us,vs,ur,vr,q,axx,axy,ayy,hdxom1,adfac,difac,dqdt)      

*     Compute small-scale velocities
!$OMP DO SCHEDULE (STATIC)      
*     U-velocity
      do j=1,nyto
*       Western boundary
        us(1,j) = 0.0d0 
*       Inner points
        do i=2,nxto
          us(i,j) = ur(i,j) - hdxom1*( axy(i,j+1) - axy(i,j) +
     &                        0.5d0*(axx(i+1,j) - axx(i-1,j)) )   
        enddo
*       Eastern boundary
        us(nxpo,j) = 0.0d0
      enddo
!$OMP END DO NOWAIT

!$OMP DO SCHEDULE (STATIC)      
*     V-velocity
      do i=1,nxto
*       Southern boundary
        vs(i,1) = 0.0d0 
*       Inner points
        do j=2,nyto
          vs(i,j) = vr(i,j) - hdxom1*( axy(i+1,j) - axy(i,j) +
     &                        0.5d0*(ayy(i,j+1) - ayy(i,j-1)) )   
        enddo
*       Northern boundary
        vs(i,nypo) = 0.0d0 
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
*         Divergences
          dqdt(i,j) = -adfac*( adfxp - adfxm + adfyp - adfym )
     &                +difac*( difxp - difxm + difyp - difym )
        enddo
      enddo
!$OMP END DO
!$OMP END PARALLEL

      END SUBROUTINE ocadiflu


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

c     ---------------------------------------------------------

      SUBROUTINE ocadblu (dbdt, b, ur, vr, axx, ayy, axy)

*     Compute the total variation of 'b' in LU formulation, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo
      USE occonst, ONLY : hdxom1,dxom2 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dbdt(nxpo,nypo)
      double precision, INTENT(IN), DIMENSION(nxpo,nypo) :: b,axy
      double precision, INTENT(IN), DIMENSION(nxpo,nyto) :: ur,axx
      double precision, INTENT(IN), DIMENSION(nxto,nypo) :: vr,ayy

*     Local variables          
      integer i,j
      double precision adfac,adfxp,adfyp,adfxm,adfym,
     &                 difac,difxp,difyp,difxm,difym,
     &                 u(nxpo,nyto),v(nxto,nypo)

*     Constant factors      
      adfac = hdxom1/6.0d0
      difac = dxom2/8.0d0

*     Boundary values are not account      
      dbdt = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,adfxp,adfxm,adfyp,adfym,
!$OMP&                      difxp,difxm,difyp,difym)
!$OMP&         SHARED  (u,v,ur,vr,b,axx,axy,ayy,adfac,difac,hdxom1,dbdt)      

*     U-velocity
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nyto
*       Western boundary
        u(1,j) = 0.0d0 
*       Inner points
        do i=2,nxto
          u(i,j) = ur(i,j) - hdxom1*( axy(i,j+1) - axy(i,j) +
     &                        0.5d0*(axx(i+1,j) - axx(i-1,j)) )   
        enddo
*       Eastern boundary
        u(nxpo,j) = 0.0d0
      enddo
!$OMP END DO NOWAIT

*     V-velocity
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxto
*       Southern boundary
        v(i,1) = 0.0d0 
*       Inner points
        do j=2,nyto
          v(i,j) = vr(i,j) - hdxom1*( axy(i+1,j) - axy(i,j) +
     &                        0.5d0*(ayy(i,j+1) - ayy(i,j-1)) )   
        enddo
*       Northern boundary
        v(i,nypo) = 0.0d0 
      enddo
!$OMP END DO

*     Advection and diffusion
!$OMP DO SCHEDULE (STATIC)      
      do j=2,nyto
        do i=2,nxto
*         **************   Small-scale advection   *************        
*         Outflux in x-axis
          adfxp = (u(i,j) + u(i+1,j) + u(i,j-1) + u(i+1,j-1)) * 
     &            (b(i,j) + b(i+1,j)) + 0.5d0*( (u(i,j) + u(i+1,j))*
     &            (b(i,j) + b(i+1,j+1)) + (u(i,j-1) + u(i+1,j-1))*
     &            (b(i,j) + b(i+1,j-1)) )
*         Influx in x-axis
          adfxm = (u(i-1,j) + u(i-1,j-1) + u(i,j-1) + u(i,j)) * 
     &            (b(i,j) + b(i-1,j)) + 0.5d0*( (u(i,j) + u(i-1,j))*
     &            (b(i,j) + b(i-1,j+1)) + (u(i-1,j-1) + u(i,j-1))*
     &            (b(i,j) + b(i-1,j-1)) )
*         Outflux in y-axis
          adfyp = (v(i-1,j+1) + v(i-1,j) + v(i,j) + v(i,j+1)) * 
     &            (b(i,j) + b(i,j+1)) + 0.5d0*( (v(i,j) + v(i,j+1))*
     &            (b(i,j) + b(i+1,j+1)) + (v(i-1,j+1) + v(i-1,j))*
     &            (b(i,j) + b(i-1,j+1)) )
*         Influx in y-axis
          adfym = (v(i-1,j) + v(i-1,j-1) + v(i,j-1) + v(i,j)) * 
     &            (b(i,j) + b(i,j-1)) + 0.5d0*( (v(i,j) + v(i,j-1))*
     &            (b(i,j) + b(i+1,j-1)) + (v(i-1,j) + v(i-1,j-1))*
     &            (b(i,j) + b(i-1,j-1)) )
*         **************   Small-scale diffusion   *************   
*         Outflux in x-axis
          difxp = (axx(i,j) + axx(i,j-1) + axx(i+1,j-1) + axx(i+1,j))*
     &            (b(i+1,j) - b(i,j)) + 0.5d0*(axy(i,j) + axy(i+1,j))*
     &            (b(i,j+1) - b(i,j-1) + b(i+1,j+1) - b(i+1,j-1))
*         Influx in x-axis
          difxm = (axx(i-1,j) + axx(i-1,j-1) + axx(i,j-1) + axx(i,j))*
     &            (b(i,j) - b(i-1,j)) + 0.5d0*(axy(i-1,j) + axy(i,j))*
     &            (b(i-1,j+1) - b(i-1,j-1) + b(i,j+1) - b(i,j-1))
*         Outflux in y-axis
          difyp = (ayy(i,j+1) + ayy(i-1,j) + ayy(i,j) + ayy(i,j+1))*
     &            (b(i,j+1) - b(i,j)) + 0.5d0*(axy(i,j+1) + axy(i,j))*
     &            (b(i+1,j+1) - b(i-1,j+1) + b(i+1,j) - b(i-1,j))
*         Influx in y-axis
          difym = (ayy(i-1,j) + ayy(i-1,j-1) + ayy(i,j-1) + ayy(i,j))*
     &            (b(i,j) - b(i,j-1)) + 0.5d0*(axy(i,j-1) + axy(i,j))*
     &            (b(i+1,j) - b(i-1,j) + b(i+1,j-1) - b(i-1,j-1))
*         Net flux
          dbdt(i,j) = -adfac*( adfxp - adfxm + adfyp - adfym ) 
     &                +difac*( difxp - difxm + difyp - difym )
        enddo
      enddo
!$OMP END DO
!$OMP END PARALLEL

      END SUBROUTINE ocadblu

#endif /* (not atmos_only) && (lu_mode) */
c
c********************************************************************
c    
#if (!defined atmos_only) && (defined use_edvel)

      SUBROUTINE ocadsed (dqdt, q, p, us, vs)

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

      END SUBROUTINE ocadsed

c     ----------------------------------------------------------

      SUBROUTINE ocadbed (dbdt, b, u, v)

*     Add a subgrid advection by eddy velocities, 
*     works for both ocean and atmos.          

*     Modules
      USE parameters, ONLY : nxto,nxpo,nyto,nypo
      USE occonst, ONLY : hdxom1 

      IMPLICIT NONE

*     I/O arguments
      double precision, INTENT(OUT) :: dbdt(nxpo,nypo)
      double precision, INTENT(IN) :: b(nxpo,nypo),
     &                                u(nxpo,nyto),v(nxto,nypo)

*     Local variables      
      integer i,j
      double precision adfac,adfxp,adfyp,adfxm,adfym

      adfac = hdxom1/6.0d0

!$OMP PARALLEL DO DEFAULT (NONE)
!$OMP&            PRIVATE (i,j,adfxp,adfxm,adfyp,adfym)
!$OMP&            SHARED  (dbdt,u,v,b,adfac)      
!$OMP&            SCHEDULE (STATIC)      
      do j=2,nyto
        do i=2,nxto
*         Outflux in x-axis
          adfxp = (u(i,j) + u(i+1,j) + u(i,j-1) + u(i+1,j-1)) * 
     &            (b(i,j) + b(i+1,j)) + 0.5d0*( (u(i,j) + u(i+1,j))*
     &            (b(i,j) + b(i+1,j+1)) + (u(i,j-1) + u(i+1,j-1))*
     &            (b(i,j) + b(i+1,j-1)) )
*         Influx in x-axis
          adfxm = (u(i-1,j) + u(i-1,j-1) + u(i,j-1) + u(i,j)) * 
     &            (b(i,j) + b(i-1,j)) + 0.5d0*( (u(i,j) + u(i-1,j))*
     &            (b(i,j) + b(i-1,j+1)) + (u(i-1,j-1) + u(i,j-1))*
     &            (b(i,j) + b(i-1,j-1)) )
*         Outflux in y-axis
          adfyp = (v(i-1,j+1) + v(i-1,j) + v(i,j) + v(i,j+1)) * 
     &            (b(i,j) + b(i,j+1)) + 0.5d0*( (v(i,j) + v(i,j+1))*
     &            (b(i,j) + b(i+1,j+1)) + (v(i-1,j+1) + v(i-1,j))*
     &            (b(i,j) + b(i-1,j+1)) )
*         Influx in y-axis
          adfym = (v(i-1,j) + v(i-1,j-1) + v(i,j-1) + v(i,j)) * 
     &            (b(i,j) + b(i,j-1)) + 0.5d0*( (v(i,j) + v(i,j-1))*
     &            (b(i,j) + b(i+1,j-1)) + (v(i-1,j) + v(i-1,j-1))*
     &            (b(i,j) + b(i-1,j-1)) )
*         Net flux
          dbdt(i,j) = -adfac*( adfxp - adfxm + adfyp - adfym )
        enddo
      enddo  
!$OMP END PARALLEL DO

      END SUBROUTINE ocadbed

#endif /* (not atmos_only) && (use_edvel) */
c
c***********************************************************************
c
      END MODULE qgosubs
c
c***********************************************************************
