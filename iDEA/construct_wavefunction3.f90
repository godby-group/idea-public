!========================================================================================!
! CONSTRUCT THE INITIAL WAVEFUNCTION FOR IMAGINARY TIME PROPAGATION (3 electrons).       !
!========================================================================================!
!
!This module contains one subroutine and one function:
!
!    construct_hamiltonian_coo (int,complex): Takes in three input arrays containing the 
!        three lowest eigenstates of a harmonic oscillator and populates one input array 
!        with the elements of the initial wavefunction.
!
!    single_index (int): Creates a single unique index for all permutations of
!        the three electron indices.
!
!----------------------------------------------------------------------------------------!
!Created by Mike Entwistle (me624@york.ac.uk)                                            !
!________________________________________________________________________________________!


!========================================================================================!
! CONSTRUCT INITIAL WAVEFUNCTION FOR IMAGINARY TIME PROPAGATION (3 electrons)            !
!----------------------------------------------------------------------------------------!
!
!Populates one input array with the elements of the initial wavefunction.
!
!    Take in one holding array and three arrays containing the pre-calculated eigenstates 
!    of a harmonic oscillator and constructs the system's initial wavefunction. The 
!    elements are then passed back into the holding array. 
!
!    Args:
!        grid (int): Spatial grid points for the system.
!        eigenstate_1,eigenstate_2,eigenstate_3 (complex, shape=grid): Three lowest 
!            eigenstates of a harmonic oscillator.
!        wavefunction (complex, shape=grid**3): Wavefunction holding array.
!
!    Returns:
!        wavefunction (complex, shape=grid**3): Populated wavefunction array.
!
!========================================================================================!
subroutine construct_wavefunction(eigenstate_1, eigenstate_2, eigenstate_3, wavefunction,&
                                  & grid)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: j, k, l, jkl
  complex (kind=dp)  :: pair_1, pair_2, pair_3

  !======================================================================================!
  ! F2PY SIGNATURE CREATION                                                              !
  !--------------------------------------------------------------------------------------!
  !f2py intent(in) :: grid
  integer :: grid

  !f2py intent(in) :: eigenstate_1
  complex (kind=dp) :: eigenstate_1(0:grid-1)

  !f2py intent(in) :: eigenstate_2
  complex (kind=dp) :: eigenstate_2(0:grid-1)

  !f2py intent(in) :: eigenstate_3
  complex (kind=dp) :: eigenstate_3(0:grid-1)

  !f2py intent(in,out) :: wavefunction
  complex (kind=dp) :: wavefunction(0:grid*grid*grid-1)
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
        do l = 0, grid-1

           !=======================================================!
           ! CALCULATE PERMUTATIONS                                !
           !=======================================================!
           pair_1 = eigenstate_1(j)*(eigenstate_2(k)*eigenstate_3(l)-eigenstate_2(l)*&
                    &eigenstate_3(k))
           pair_2 = eigenstate_2(j)*(eigenstate_3(k)*eigenstate_1(l)-eigenstate_1(k)*&
                    &eigenstate_3(l))
           pair_3 = eigenstate_3(j)*(eigenstate_1(k)*eigenstate_2(l)- eigenstate_2(k)*&
                    &eigenstate_1(l))
           !_______________________________________________________!

           !=======================================================!
           ! WAVEFUNCTION ELEMENT                                  !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           wavefunction(jkl) = pair_1 + pair_2 + pair_3
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

end subroutine construct_wavefunction
!_______________________________________________________!


!========================================================================================!
! SINGLE UNIQUE INDEX RETURN (RETURN: INT)                                               !
!----------------------------------------------------------------------------------------!
! Takes every permutation of the three electron indices and creates a single unique      !
! index.                                                                                 ! 
!========================================================================================!
function single_index(j, k, l, grid) result(z)

  implicit none

  integer,intent(in)  :: j, k, l, grid
  integer :: z

  z = l + k*grid + j*grid**2

  return

end function single_index
!________________________________________________________________________________________!
