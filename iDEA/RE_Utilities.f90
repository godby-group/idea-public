!========================================================================================!
! REVERSE ENGINEERING: UTILITIES                                                         !
!========================================================================================!
!
!This module contains one subroutines:
!
!    continuity_eqn (int, real): Calculates the electron current density of the system   
!        for a particular time-step by solving the continuity equation.
!                      
!----------------------------------------------------------------------------------------!
!Created by Mike Entwistle, Ed Higgins, Aaron Hopkinson and Matthew Smith                !
!________________________________________________________________________________________!


!========================================================================================!
! CONTINUITY EQUATION (RETURN: REAL ARRAY)                                               !
!----------------------------------------------------------------------------------------!
!
!Returns a real array containing the electron current density at time t, calculated 
!through solving the continuity equation.
!
!    Args:
!        grid (int): Spatial grid points for the system.
!        deltax (real): Distance between spatial grid points.
!        deltat (real): Distance between temporal grid points.
!        density_new (real, shape=grid): Array containing electron density at time t.
!        density_old (real, shape=grid): Array containing electron density at time t-dt.
!        current_density(real, shape=grid): Array to be populated with the electron
!            current density at time t.
!
!    Returns:
!        current_density (real, shape=grid): Array containing electron current density at
!            time t.
!
!========================================================================================!
subroutine continuity_eqn(current_density, density_new, density_old, deltax, deltat, grid)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: j, k
  real(kind=dp)      :: prefactor
  
  !======================================================================================!
  ! F2PY SIGNATURE CREATION                                                              !
  !--------------------------------------------------------------------------------------! 
  !f2py intent(in) :: grid
  integer :: grid

  !f2py intent(in) :: deltax
  real (kind=dp) :: deltax

  !f2py intent(in) :: deltat
  real (kind=dp) :: deltat

  !f2py intent(in) :: density_new, density_old
  real (kind=dp) :: density_new(0:grid-1), density_old(0:grid-1)

  !f2py intent(in, out) :: current_density
  real (kind=dp) :: current_density(0:grid-1)
  !______________________________________________________________________________________!

  ! Parameter and array initialisations
  prefactor = deltax/deltat

  !=============================================================!
  ! MAIN PROGRAM LOOP                                           !
  !=============================================================!
  do j=1,grid-1
     do k=0, j-1
        current_density(j) = current_density(j) - prefactor*(density_new(k)-density_old(k))
     end do
  end do
  !_____________________________________________________________!

end subroutine continuity_eqn
!________________________________________________________________________________________!

