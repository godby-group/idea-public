!==============================================================================!
! CONSTRUCT THE SYSTEMS HAMILTONIAN OPERATOR IN COOrdinate FORM (2 electrons). !
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!
  ! Compile with:
  ! f2py -c -m create_hamiltonian_coo create_hamiltonian_coo.f90 --opt'-march=native -O3'
!==============================================================================!
!
!This module contains two functions:
!
!    gind (int): Creates a unique index for all combinations of electron
!        spatial combinations.
!
!    create_hamiltonian_coo (int,complex): Populates three input arrays with a
!        COOrdinate form of the systems Hamiltonian. These can then be
!        constructed into a COOrdinate sparse matrix.
!
!   * TODO: openMP parallelization for the main loop.
!   * TODO: Change algorithm to remove IF conditionals for non-halo elements.
!   * TODO: Test if compiling with mkl gives a performance increase.
!   * TODO: Pre-calc r and pass as array.
!
!------------------------------------------------------------------------------!
!Created by Aaron James Long (al876@york.ac.uk)
!______________________________________________________________________________!


!==============================================================================!
! CREATE HAMILTONIAN OPERATOR IN SPARSE COORDINATE FORM (2 electrons)          !
!------------------------------------------------------------------------------!
!
!Populates three input arrays with a COOrdinate form of the systems Hamiltonian.
!
!    Take in 3 holding arrays and one array containing the pre-calculated
!    diagonals and construct the systems Hamiltonian. Coordinates and data
!    are then passed back in the holding arrays, these can then be converted
!    into a sparse COOrdinate matrix (efficient storage).
!
!    NB: As we iterate through each non-sparse element, we increase our i-th
!        element counter (i) corresponding to the total number of non-sparse
!        elements (~max_size).
!
!    Args:
!        max_size (int, optional): Estimate for the totnal number of non-sparse
!            elements in the Hamiltonian.
!        v_size (int, optional): Size of the Hamiltonian diagonals array.
!        jmax, kmax (int): Spatial grid points for the system.
!        coo_j,coo_k (int, shape=max_size): Coordinate holding arrays.
!        coo_data (complex, shape=max_size): Data for a given matrix element.
!        r (real): Potential spatial component.
!
!    Returns:
!        coo_j,coo_k (int, shapse=max_size): Populated coordinate arrays.
!        coo_data (complex, shape=max_size): Populated data array.
!
!==============================================================================!
subroutine create_hamiltonian_coo(coo_j, coo_k, coo_data, h_diagonals,&
                                  & r, jmax, kmax,  max_size, v_size)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: i, j, k, jk, v_size

  !=============================================================================!
  ! F2PY SIGNATURE CREATION                                                     !
  !-----------------------------------------------------------------------------!
  !f2py intent(in) :: r
  real (kind=dp) :: r

  !f2py intent(in) :: jmax, kmax, max_size
  integer :: jmax, kmax, max_size

  !f2py intent(in, out) :: coo_j
  !f2py integer,intent(hide),depend(coo_x) :: max_size=shape(coo_j, 0)
  integer :: coo_j(0:max_size-1)

  !f2py intent(in, out) :: coo_k
  integer :: coo_k(0:max_size-1)

  !f2py intent(in, out) :: coo_data
  complex (kind=dp) :: coo_data(0:max_size-1)

  !f2py intent(in) :: h_diagonals
  !f2py integer,intent(hide),depend(h_diagonals) :: v_size=shape(h_diagonals, 0)
  complex (kind=dp) :: h_diagonals(0:v_size-1)
  !____________________________________________________________________________!


  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external                  :: gind              !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!

  ! Counter for ith element of the COO arrays.
  i = 0

  do j = 0, jmax-1
     do k = 0, kmax-1

        !=======================================================!
        ! ASSIGN DIAGONAL                                       !
        !=======================================================!
        jk = gind(j, k, jmax)
        coo_j(i) = jk
        coo_k(i) = jk
        coo_data(i) = h_diagonals(jk)
        i = i+1
        !_______________________________________________________!

        !=======================================================!
        ! ASSIGN ADDITIONAL BANDS                               !
        !=======================================================!
        if (j < jmax-1) then
           coo_j(i) = jk
           coo_k(i) = gind(j+1, k, jmax)
           coo_data(i) = -r
           i = i+1
        end if

        if (j > 0) then
           coo_j(i) = jk
           coo_k(i) = gind(j-1, k, jmax)
           coo_data(i) = -r
           i = i+1
        end if

        if (k < kmax-1) then
           coo_j(i) = jk
           coo_k(i) = gind(j, k+1, kmax)
           coo_data(i) = -r
           i = i+1
        end if

        if (k > 0) then
           coo_j(i) = jk
           coo_k(i) = gind(j, k-1, kmax)
           coo_data(i) = -r
           i = i+1
        end if
        !_______________________________________________________!


     end do
  end do
  !_______________________________________________________!

end subroutine create_hamiltonian_coo
!_______________________________________________________!


!==============================================================================!
! FLATTENED ARRAY ELEMENT RETURN (RETURN: INT)                                 !
!------------------------------------------------------------------------------!
! Here we pass current element indices for a matrix and return a single index  !
! relating to a flattened array of the matrix.                                 !
!==============================================================================!
function gind(j, k, jmax) result(z)

  implicit none

  integer,intent(in)  :: j, k, jmax
  integer :: z

  z = k + j*jmax

  return

end function gind
!______________________________________________________________________________!
