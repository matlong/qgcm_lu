
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
      MODULE atedvl

      USE parameters, ONLY : nxpa,nypa,nxta,nyta,nla 

      IMPLICIT NONE

      PUBLIC
      SAVE

#if (defined use_edvel) && (!defined ocean_only)      
      double precision uaed(nxpa,nyta,nla),vaed(nxta,nypa,nla)
#endif  

      END MODULE atedvl
c
c***********************************************************************

