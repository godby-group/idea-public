!========================================================================================!
! CONSTRUCT THE INITIAL WAVEFUNCTION FOR IMAGINARY TIME PROPAGATION (3 electrons).       !
!========================================================================================!
!
!This module contains one subroutine:
!
!    construct_hamiltonian_coo (int,complex): Takes in three input arrays containing the 
!        three lowest eigenstates of a harmonic oscillator and populates one input array 
!        with the elements of the initial wavefunction.
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
!        wavefunction (complex, shape=~(grid**3)/6): Wavefunction holding array.
!
!    Returns:
!        wavefunction (complex, shape=~(grid**3)/6): Populated wavefunction array.
!
!========================================================================================!
subroutine construct_wavefunction(eigenstate_1, eigenstate_2, eigenstate_3, wavefunction,&
                                  & grid)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: i, j, k, l, jkl
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
  complex (kind=dp) :: wavefunction(0:((grid*(grid+1)*(grid+2))/6)-1)
  !______________________________________________________________________________________!


  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external :: single_index                       !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!
  i = 0
  do j = 0, grid-1
     do k = 0, j
        do l = 0, k

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
           wavefunction(i) = pair_1 + pair_2 + pair_3
           i = i+1
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

end subroutine construct_wavefunction
!_______________________________________________________!
