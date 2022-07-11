c***********************************************************************
c     Q-GCM Version 1.5.0 : last modified 02/07/2013
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
      MODULE luparam

*     Contains all array dimensioning parameter of LU variables.

      USE parameters

      IMPLICIT NONE

      PUBLIC
      SAVE

#ifdef lu_mode
      integer seed,ncore
      
      integer :: nmo,neofo,nposub,npores
      PARAMETER ( nmo = 50, neofo = nlo*(nxpo*nyto+nxto*nypo) )

      integer :: nma,neofa,npasub,npares
      PARAMETER ( nma = 50, neofa = nla*(nxpa*nyta+nxta*nypa) )
#endif

      END MODULE luparam
c
c***********************************************************************
