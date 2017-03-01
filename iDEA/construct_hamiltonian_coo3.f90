!==============================================================================!
! CONSTRUCT THE SYSTEMS HAMILTONIAN OPERATOR IN COOrdinate FORM (3 electrons). !
!==============================================================================!
!
!This module contains one subroutine and one function:
!
!    construct_hamiltonian_coo (int,complex): Populates three input arrays with 
!        a COOrdinate form of the systems Hamiltonian. These can then be 
!        constructed into a COOrdinate sparse matrix.
!
!    single_index (int): Creates a single unique index for all permutations of
!        the three electron indices.
!
!   * TODO: openMP parallelization for the main loop.
!   * TODO: Test if compiling with mkl gives a performance increase.
!
!------------------------------------------------------------------------------!
!Created by Mike Entwistle (me624@york.ac.uk)                                  !
!______________________________________________________________________________!


!==============================================================================!
! CONSTRUCT HAMILTONIAN OPERATOR IN SPARSE COOrdinate FORM (3 electrons)       !
!------------------------------------------------------------------------------!
!
!Populates three input arrays with a COOrdinate form of the systems Hamiltonian.
!
!    Take in 3 holding arrays and one array containing the pre-calculated
!    diagonals and construct the system's Hamiltonian. Coordinates and data are
!    then passed back into the holding arrays. These can then be converted into 
!    a COOrdinate sparse matrix (efficient storage).
!
!    NB: As we iterate through each non-zero element, we increase our i-th
!        element counter (i) corresponding to the total number of non-zero
!        elements (~max_size).
!
!    Args:
!        max_size (int, optional): Estimate for the total number of non-zero
!            elements in the Hamiltonian.
!        v_size (int, optional): Size of the Hamiltonian diagonals array. 
!        grid (int): Spatial grid points for the system.
!        coo_1,coo_2 (int, shape=max_size): Coordinate holding arrays.
!        coo_data (complex, shape=max_size): Data for a given matrix element.
!        r (complex): Potential spatial component.
!
!    Returns:
!        coo_1,coo_2 (int, shape=max_size): Populated coordinate arrays.
!        coo_data (complex, shape=max_size): Populated data array.
!
!==============================================================================!
subroutine construct_hamiltonian_coo(coo_1, coo_2, coo_data, h_diagonals, r,&
                                     & grid, max_size, v_size)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: i, j, k, l, jkl, diag, v_size

  !============================================================================!
  ! F2PY SIGNATURE CREATION                                                    !
  !----------------------------------------------------------------------------!
  !f2py intent(in) :: r
  complex (kind=dp) :: r

  !f2py intent(in) :: grid, max_size
  integer :: grid, max_size

  !f2py intent(in, out) :: coo_1
  !f2py integer,intent(hide),depend(coo_1) :: max_size=shape(coo_1, 0)
  integer :: coo_1(0:max_size-1)

  !f2py intent(in, out) :: coo_2
  integer :: coo_2(0:max_size-1)

  !f2py intent(in, out) :: coo_data
  complex (kind=dp) :: coo_data(0:max_size-1)

  !f2py intent(in) :: h_diagonals
  !f2py integer,intent(hide),depend(h_diagonals) :: v_size=shape(h_diagonals, 0)
  complex (kind=dp) :: h_diagonals(0:v_size-1)
  !____________________________________________________________________________!


  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external :: single_index                       !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!

  ! Counter for ith element of the COO arrays.
  i = 0
  !Counter for main diagonal
  diag = 0

  do j = 1, grid-2
     do k = 1, grid-2
        do l = 1, grid-2

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = h_diagonals(diag)
           i = i+1
           diag = diag+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           coo_1(i) = single_index(j+1, k, l, grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1

           coo_1(i) = single_index(j-1, k, l,  grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1

           coo_1(i) = single_index(j, k+1, l, grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1

           coo_1(i) = single_index(j, k-1, l, grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1

           coo_1(i) = single_index(j, k, l+1, grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1

           coo_1(i) = single_index(j, k, l-1, grid)
           coo_2(i) = jkl
           coo_data(i) = -r
           i = i+1
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do j = 0, grid-1, grid-1
     do k = 0, grid-1
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = h_diagonals(diag)
           diag = diag+1
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           if (j < grid-1) then
              coo_1(i) = single_index(j+1, k, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (j > 0) then
              coo_1(i) = single_index(j-1, k, l,  grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k < grid-1) then
              coo_1(i) = single_index(j, k+1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k > 0) then
              coo_1(i) = single_index(j, k-1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (l < grid-1) then 
              coo_1(i) = single_index(j, k, l+1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if 

           if (l > 0) then
              coo_1(i) = single_index(j, k, l-1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do k = 0, grid-1, grid-1
     do j = 1, grid-2
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = h_diagonals(diag)
           diag = diag+1
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           if (j < grid-1) then
              coo_1(i) = single_index(j+1, k, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (j > 0) then
              coo_1(i) = single_index(j-1, k, l,  grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k < grid-1) then
              coo_1(i) = single_index(j, k+1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k > 0) then
              coo_1(i) = single_index(j, k-1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (l < grid-1) then 
              coo_1(i) = single_index(j, k, l+1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if 

           if (l > 0) then
              coo_1(i) = single_index(j, k, l-1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do l = 0, grid-1, grid-1
     do j = 1, grid-2
        do k = 1, grid-2

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = h_diagonals(diag)
           diag = diag+1
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           if (j < grid-1) then
              coo_1(i) = single_index(j+1, k, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (j > 0) then
              coo_1(i) = single_index(j-1, k, l,  grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k < grid-1) then
              coo_1(i) = single_index(j, k+1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (k > 0) then
              coo_1(i) = single_index(j, k-1, l, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if

           if (l < grid-1) then 
              coo_1(i) = single_index(j, k, l+1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if 

           if (l > 0) then
              coo_1(i) = single_index(j, k, l-1, grid)
              coo_2(i) = jkl
              coo_data(i) = -r
              i = i+1
           end if  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

end subroutine construct_hamiltonian_coo
!_______________________________________________________!


!==============================================================================!
! SINGLE UNIQUE INDEX RETURN (RETURN: INT)                                     !
!------------------------------------------------------------------------------!
! Takes every permutation of the three electron indices and creates a single   !
! unique index.                                                                !    
!==============================================================================!
function single_index(j, k, l, grid) result(z)

  implicit none

  integer,intent(in)  :: j, k, l, grid
  integer :: z

  z = l + k*grid + j*grid**2

  return

end function single_index
!______________________________________________________________________________!
