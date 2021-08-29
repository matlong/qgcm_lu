c**************************************************************************
c
c     This module includes all subroutines of filtering for performing
c     the program coarse_grain.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************      
c
      MODULE coarse_subs

      IMPLICIT NONE
      PRIVATE

*     Subroutines      
      PUBLIC :: gaussf, pfilt, pfilt_cycl, p2uv

      CONTAINS

c**************************************************************************      
c
      SUBROUTINE gaussf (kerl, wt, wd)

*     Construct 2D Gaussian filter kernel kerl of width wd. 
*     The total weight wt is also calculated.

      IMPLICIT NONE    
      
*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(OUT) :: kerl(wd+1,wd+1),wt

*     Local variables      
      integer i,j,c
      double precision pi

      wt = 0.0d0
      pi = 3.14159265358979324D0
      c = wd/2 + 1 

      do i=1,wd+1  
        do j=1,wd+1
*         Gaussian convolution kernel        
          kerl(i,j) = ( 6.0d0/(pi*(wd**2)) ) *
     &                exp( - 6.0d0*( (i-c)**2 + (j-c)**2 ) / (wd**2) ) 
*         Total weight of kernel          
          wt = wt + kerl(i,j)
        enddo
      enddo

      END SUBROUTINE gaussf
c
c**************************************************************************
c      
      SUBROUTINE pfilt (pf, pr, p, fk, wt, wd)

*     Coarse-graining of eddy-resolving snapshot.

*     Modules          
      USE param, ONLY : nxpo,nypo,nxpoc,nypoc,nso

      IMPLICIT NONE

*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(IN) :: fk(wd+1,wd+1),wt,p(nxpo,nypo)
      double precision, INTENT(OUT), DIMENSION(nxpoc,nypoc) :: pf,pr
*
*     fk is filter kernel of total weight wt and width wd
*     p is eddy-resolving presure (m^2/s^2)
*     pf and pr are corase-grained and residual pressure (m^2/s^2)

*     Local variables
      integer nh,i,j,ii,jj,il,jl,ig,jg
      double precision psum,wsum

      nh = wd/2 + 1 !! size of half-kernel

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,ii,jj,il,jl,ig,jg,psum,wsum)
!$OMP&         SHARED  (p,pf,pr,fk,wt,wd,nh)

*     Filtering inner points     
*     ----------------------
!$OMP DO SCHEDULE (STATIC)      
      do i=2,nxpoc-1 !! coarse-grid index
        ii = 1 + (i-1)*nso !! fine-grid index
        do j=2,nypoc-1
          jj = 1 + (j-1)*nso
          psum = 0.0d0 !! convoluted state
          wsum = 0.0d0 !! accounted weight
          do il=1,wd+1 !! local index (within kernel)
            ig = ii + il - nh !! global index (fine-grid)
            if ( (ig.lt.1) .or. (ig.gt.nxpo) ) then
              cycle !! skip iter.
            endif
            do jl=1,wd+1
              jg = jj + jl - nh
              if ( (jg.lt.1) .or. (jg.gt.nypo) ) then
                cycle 
              endif  
*             Convolution of p by fk
              psum = psum + fk(il,jl)*p(ig,jg)
              wsum = wsum + fk(il,jl)            
            enddo
          enddo
*         Rescale pf by wt          
          pf(i,j) = psum*wt/wsum
*         Substract pr
          pr(i,j) = p(ii,jj) - pf(i,j)
        enddo
      enddo
!$OMP END DO NOWAIT      

*     Filling boundary values
*     -----------------------     
*     Update W/E value      
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypoc
        jj = 1 + (j-1)*nso
        pf(1,j) = p(1,jj)*wt
        pr(1,j) = p(1,jj) - pf(1,j)
        pf(nxpoc,j) = p(nxpo,jj)*wt
        pr(nxpoc,j) = p(nxpo,jj) - pf(nxpoc,j)
      enddo
!$OMP END DO NOWAIT     
*     Update N/S value
!$OMP DO SCHEDULE (STATIC)      
      do i=2,nxpoc-1
        ii = 1 + (i-1)*nso
        pf(i,1) = p(ii,1)*wt 
        pr(i,1) = p(ii,1) - pf(i,1)
        pf(i,nypoc) = p(ii,nypo)*wt
        pr(i,nypoc) = p(ii,nypo) - pf(i,nypoc)
      enddo
!$OMP END DO      
!$OMP END PARALLEL

      END SUBROUTINE pfilt
c
c**************************************************************************
c      
      SUBROUTINE pfilt_cycl (pf, pr, p, fk, wt, wd)

*     Coarse-graining of eddy-resolving snapshot in the cyclic channel.

*     Modules          
      USE param, ONLY : nxpo,nypo,nxpoc,nypoc,nso

      IMPLICIT NONE

*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(IN) :: fk(wd+1,wd+1),wt,p(nxpo,nypo)
      double precision, INTENT(OUT), DIMENSION(nxpoc,nypoc) :: pf,pr
*
*     fk is filter kernel of total weight wt and width wd
*     p is eddy-resolving presure (m^2/s^2)
*     pf and pr are corase-grained and residual pressure (m^2/s^2)

*     Local variables
      integer nh,i,j,ii,jj,il,jl,ig,jg
      double precision psum,wsum

      nh = wd/2 + 1 !! size of half-kernel

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,ii,jj,il,jl,ig,jg,psum,wsum)
!$OMP&         SHARED  (p,pf,pr,fk,wt,wd,nh)

*     Filtering inner points     
*     ----------------------
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc-1 !! coarse-grid index
        ii = 1 + (i-1)*nso !! fine-grid index
        do j=2,nypoc-1
          jj = 1 + (j-1)*nso
          psum = 0.0d0 !! convoluted state
          wsum = 0.0d0 !! accounted weight
          do jl=1,wd+1 !! local index (within kernel)
            jg = jj + jl - nh !! global index (fine-grid)
            if ( (jg.lt.1) .or. (jg.gt.nypo) ) then
              cycle !! skip iter.
            endif
            do il=1,wd+1
              ig = ii + il - nh
              if (jg.lt.1) then
                  psum = psum + fk(il,jl)*p(ig+nxpo-1,jg)
              else if (jg.gt.nypo) then
                  psum = psum + fk(il,jl)*p(ig-nxpo+1,jg)
              else
                  psum = psum + fk(il,jl)*p(ig,jg)
              endif  
              wsum = wsum + fk(il,jl)            
            enddo
          enddo
*         Rescale pf by wt          
          pf(i,j) = psum*wt/wsum
*         Substract pr
          pr(i,j) = p(ii,jj) - pf(i,j)
        enddo
      enddo
!$OMP END DO NOWAIT      

*     Filling boundary values
*     -----------------------     
*     Update N/S value
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc
        ii = 1 + (i-1)*nso
        pf(i,1) = p(ii,1)*wt 
        pr(i,1) = p(ii,1) - pf(i,1)
        pf(i,nypoc) = p(ii,nypo)*wt
        pr(i,nypoc) = p(ii,nypo) - pf(i,nypoc)
      enddo
!$OMP END DO NOWAIT     
*     Update W/E value      
!$OMP DO SCHEDULE (STATIC)      
      do j=2,nypoc-1
        pf(nxpoc,j) = pf(1,j)
        pr(nxpoc,j) = pr(1,j)
      enddo
!$OMP END DO      
!$OMP END PARALLEL

      END SUBROUTINE pfilt_cycl
c
c**************************************************************************      
c
      SUBROUTINE p2uv (u, v, p, rdxf0)

*     Compute geostrophic velocities from pressure

*     Modules
      USE param

      IMPLICIT NONE    
      
*     I/O arguments
      double precision, INTENT(OUT) :: u(nxpoc,nytoc),v(nxtoc,nypoc)
      double precision, INTENT(IN) :: p(nxpoc,nypoc),rdxf0

*     Local variables      
      integer i,j

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j)
!$OMP&         SHARED  (u,v,p,rdxf0)
      
*     Zonal velocity  
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc
        do j=1,nytoc
          u(i,j) = -rdxf0*( p(i,j+1) - p(i,j) )
        enddo
      enddo
!$OMP END DO NOWAIT      
   
*     Meridional velocity      
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypoc
        do i=1,nxtoc
          v(i,j) =  rdxf0*( p(i+1,j) - p(i,j) )
        enddo
      enddo
!$OMP END DO 
!$OMP END PARALLEL

      END SUBROUTINE p2uv
c      
c**************************************************************************      
c
      END MODULE coarse_subs    
c      
c**************************************************************************
