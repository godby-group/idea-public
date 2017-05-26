!========================================================================================!
! REVERSE ENGINEERING: UTILITIES                                                         !
!========================================================================================!
!
!This module contains two subroutines:
!
!    continuity_eqn (int, real): Calculates the electron current density of the system   
!        for a particular time-step by solving the continuity equation.
!
!    compare (int, real): Checks whether the Kohn-Sham electron current density matches 
!        the exact electron current density. 
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


!========================================================================================!
! COMPARE (RETURN: LOGICAL)                                                              !
!----------------------------------------------------------------------------------------!
!                                                                                        
!Checks whether the Kohn-Sham electron current density matches the exact electron current 
!density.
!
!    Args:
!        grid (int): Spatial grid points for the system.
!        current_density_ks (real, shape=grid): Kohn-Sham electron current density. 
!        current_density_ext (real, shape=grid): Exact electron current density.
!        tolerance (real): Allowed tolerance between the Kohn-Sham electron current 
!            density and the exact electron current density.   
!
!    Returns:    
!        current_density_check (logical): Does the Kohn-Sham electron current density 
!            match the exact electron current density within the specified tolerance?
!                                                                                    
!========================================================================================!
subroutine compare(current_density_ks, current_density_ext, tolerance, grid)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  
  !======================================================================================!
  !F2PY SIGNATURE CREATION                                                               !
  !--------------------------------------------------------------------------------------!
  !f2py intent(in) :: grid
  integer :: grid

  !f2py intent(in) :: tolerance
  real (kind=dp) :: tolerance

  !f2py intent(in) :: current_density_ext, current_density_ks
  real (kind=dp) :: current_density_ext(0:grid-1), current_density_ks(0:grid-1)

  !f2py intent(out) :: current_density_check
  logical :: current_density_check
  !______________________________________________________________________________________!

  current_density_check = all(abs(current_density_ks-current_density_ext) < tolerance)

end subroutine compare
!________________________________________________________________________________________!
