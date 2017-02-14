!========================================================================================!
! CONSTRUCT THE INITIAL WAVEFUNCTION FOR IMAGINARY TIME PROPAGATION (2 electrons).       !
!========================================================================================!
!
!This module contains one subroutine and one function:
!
!    construct_hamiltonian_coo (int,complex): Takes in two input arrays containing the  
!        two lowest eigenstates of a harmonic oscillator and populates one input array 
!        with the elements of the initial wavefunction.
!
!    single_index (int): Creates a single unique index for all permutations of
!        the two electron indices.
!
!----------------------------------------------------------------------------------------!
!Created by Mike Entwistle (me624@york.ac.uk)                                            !
!________________________________________________________________________________________!


!========================================================================================!
! CONSTRUCT INITIAL WAVEFUNCTION FOR IMAGINARY TIME PROPAGATION (2 electrons)            !
!----------------------------------------------------------------------------------------!
!
!Populates one input array with the elements of the initial wavefunction.
!
!    Take in one holding array and two arrays containing the pre-calculated eigenstates 
!    of a harmonic oscillator and constructs the system's initial wavefunction. The 
!    elements are then passed back into the holding array. 
!
!    Args:
!        grid (int): Spatial grid points for the system.
!        eigenstate_1,eigenstate_2 (complex, shape=grid): Three lowest eigenstates of a
!            harmonic oscillator.
!        wavefunction (complex, shape=grid**2): Wavefunction holding array.
!
!    Returns:
!        wavefunction (complex, shape=grid**2): Populated wavefunction array.
!
!========================================================================================!
subroutine construct_wavefunction(eigenstate_1, eigenstate_2, wavefunction, grid)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: j, k, jk
  complex (kind=dp)  :: pair

  !======================================================================================!
  ! F2PY SIGNATURE CREATION                                                              !
  !--------------------------------------------------------------------------------------!
  !f2py intent(in) :: grid
  integer :: grid

  !f2py intent(in) :: eigenstate_1
  complex (kind=dp) :: eigenstate_1(0:grid-1)

  !f2py intent(in) :: eigenstate_2
  complex (kind=dp) :: eigenstate_2(0:grid-1)

  !f2py intent(in,out) :: wavefunction
  complex (kind=dp) :: wavefunction(0:grid*grid-1)
  !______________________________________________________________________________________!


  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external :: single_index                       !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!

  do j = 0, grid-1
     do k = 0, grid-1

           !=======================================================!
           ! CALCULATE PERMUTATIONS                                !
           !=======================================================!
           pair = eigenstate_1(j)*eigenstate_2(k) - eigenstate_1(k)*eigenstate_2(j)
           !_______________________________________________________!

           !=======================================================!
           ! WAVEFUNCTION ELEMENT                                  !
           !=======================================================!
           jk = single_index(j, k, grid)
           wavefunction(jk) = pair
           !_______________________________________________________!

     end do
  end do
  !_______________________________________________________!

end subroutine construct_wavefunction
!_______________________________________________________!


!========================================================================================!
! SINGLE UNIQUE INDEX RETURN (RETURN: INT)                                               !
!----------------------------------------------------------------------------------------!
! Takes every permutation of the two electron indices and creates a single unique index. !                                                                                ! 
!========================================================================================!
function single_index(j, k, grid) result(z)

  implicit none

  integer,intent(in)  :: j, k, grid
  integer :: z

  z = k + j*grid 

  return

end function single_index
!________________________________________________________________________________________!
