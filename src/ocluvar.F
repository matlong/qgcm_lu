c***********************************************************************
c     Q-GCM Version 1.5.0 : last modified 21/06/2013
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
      MODULE ocluvar

*     Allocate the location uncertainty ocean variables in Q-GCM_LU model, 
*     including the spatial modes of random velocity and the diffusion tensor.
      
      USE parameters, ONLY : nxpo,nypo,nxto,nyto,nlo
#ifdef lu_mode      
      USE luparam, ONLY : nmo,neofo
#endif

      IMPLICIT NONE

      PUBLIC
      SAVE

#if (defined lu_mode) && (!defined atmos_only)      
      double precision eofoc(neofo,nmo),uro(nxpo,nyto,nlo),
     &                 vro(nxto,nypo,nlo),aoxx(nxpo,nyto,nlo),
     &                 aoyy(nxto,nypo,nlo),aoxy(nxpo,nypo,nlo) 
#  ifdef gmlu_param      
      double precision apoxx(nxpo,nyto,nlo),apoyy(nxto,nypo,nlo),
     &                 apoxy(nxpo,nypo,nlo),keopod(nlo)
#  endif /* gmlu_param */      
*     uro, vro are components of the oceanic velocity noise      
*     aoxx, aoxy, aoyy are components of the horizontal diffusion tensor
#endif      

      END MODULE ocluvar
c
c***********************************************************************

