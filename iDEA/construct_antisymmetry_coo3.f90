!========================================================================================!
! CONSTRUCT THE REDUCTION AND EXPANSION MATRICES IN COOrdinate FORM (3 electrons).       !  
!========================================================================================!
!
!This module contains one subroutine and one function:
!
!    construct_antisymmetry_coo (int,real): Populates three input arrays with a 
!        COOrdinate form of the reduction and expansion matrices. These can then be 
!        constructed into a COOrdinate sparse matrix.
!
!    single_index (int): Creates a single unique index for all permutations of the three 
!        electron indices.
!
!----------------------------------------------------------------------------------------!
!Created by Mike Entwistle (me624@york.ac.uk)                                            !
!________________________________________________________________________________________!


!========================================================================================!
! CONSTRUCT THE REDUCTION AND EXPANSION MATRICES IN SPARSE COORDINATE FORM (3 electrons) !         
!----------------------------------------------------------------------------------------!
!
!Populates three input arrays with a COOrdinate form of the reduction and expansion
!matrices.
!
!    Take in 3 holding arrays and construct the reduction and expansion matrices. 
!    Coordinates and data are then passed back into the holding arrays. These can then be 
!    converted into a COOrdinate sparse matrix (efficient storage).
!
!    Args:
!        coo_size (int): Total number of non-zero elements in the reduction 
!            matrix (and ~ one sixth the total number of non-zero elements in the 
!            expansion matrix)
!        grid (int): Spatial grid points of the system.
!        coo_1,coo_2 (int, shape=coo_size): Coordinate holding arrays for the reduction 
!            matrix.
!        coo_3,coo_4 (int, shape=grid**3): Coordinate holding arrays for the expansion 
!            matrix.
!        coo_data_1 (real, shape=coo_size): Data for a given matrix element in the 
!            reduction matrix.
!        coo_data_2 (real, shape=grid**3): Data for a given matrix element in the 
!            expansion matrix. 
!
!    Returns:
!        coo_1,coo_2 (int, shape=coo_size): Populated coordinate arrays for the reduction
!            matrix.
!        coo_3,coo_4 (int, shape=grid**3): Populated coordinate arrays for the expansion 
!            matrix.
!        coo_data_1 (real, shape=coo_size): Populated data array for the reduction matrix.
!        coo_data_2 (real, shape=grid**3): Populated data array for the expansion matrix.
!
!========================================================================================!
subroutine construct_antisymmetry_coo(coo_1, coo_2, coo_3, coo_4, coo_data_1, coo_data_2,&
                                      & grid, coo_size)

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: i_plus, i_minus, j, k, l, jkl, jlk, klj, kjl, ljk, lkj

  !======================================================================================!
  ! F2PY SIGNATURE CREATION                                                              !
  !--------------------------------------------------------------------------------------!
  !f2py intent(in) :: grid, coo_size
  integer :: grid, coo_size

  !f2py intent(in, out) :: coo_1
  integer :: coo_1(0:coo_size-1)

  !f2py intent(in, out) :: coo_2
  integer :: coo_2(0:coo_size-1)

  !f2py intent(in, out) :: coo_3
  integer :: coo_3(0:grid*grid*grid-1)

  !f2py intent(in, out) :: coo_4
  integer :: coo_4(0:grid*grid*grid-1)

  !f2py intent(in, out) :: coo_data_1
  real (kind=dp) :: coo_data_1(0:coo_size-1)

  !f2py intent(in, out) :: coo_data_2
  real (kind=dp) :: coo_data_2(0:grid*grid*grid-1)
  !______________________________________________________________________________________!

 
  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external :: single_index                       !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!

  ! Counter for ith element of the COO arrays.
  i_plus = 0
  i_minus = 0

  do j = 0, grid-1
     do k = 0, j
        do l = 0, k
        
           !=======================================================!
           ! ASSIGN REDUCTION MATRIX ELEMENT                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i_plus) = i_plus
           coo_2(i_plus) = jkl
           coo_data_1(i_plus) = 1.0_dp
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN EXPANSION MATRIX ELEMENTS                      !
           !=======================================================!
           coo_3(i_plus) = jkl
           coo_4(i_plus) = i_plus
           coo_data_2(i_plus) = 1.0_dp

           if((j /= k) .AND. (k == l)) then
               kjl = single_index(k, j, l, grid)
               coo_3(grid**3-1-i_minus) = kjl
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1

               lkj = single_index(l, k, j, grid)
               coo_3(grid**3-1-i_minus) = lkj
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1
           end if 

           if((j == k) .AND. (k /= l)) then
               jlk = single_index(j, l, k, grid)
               coo_3(grid**3-1-i_minus) = jlk
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1

               lkj = single_index(l, k, j, grid)
               coo_3(grid**3-1-i_minus) = lkj
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1
           end if

           if((j /= k) .AND. (k /= l)) then
               jlk = single_index(j, l, k, grid)
               coo_3(grid**3-1-i_minus) = jlk
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1

               kjl = single_index(k, j, l, grid)
               coo_3(grid**3-1-i_minus) = kjl
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1

               lkj = single_index(l, k, j, grid)
               coo_3(grid**3-1-i_minus) = lkj
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = -1.0_dp
               i_minus = i_minus+1

               klj = single_index(k, l, j, grid)
               coo_3(grid**3-1-i_minus) = klj
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = 1.0_dp
               i_minus = i_minus+1

               ljk = single_index(l, j, k, grid)
               coo_3(grid**3-1-i_minus) = ljk
               coo_4(grid**3-1-i_minus) = i_plus
               coo_data_2(grid**3-1-i_minus) = 1.0_dp
               i_minus = i_minus+1
           end if

           !_______________________________________________________!

           i_plus = i_plus+1

        end do 
     end do
  end do

  !_______________________________________________________!

end subroutine construct_antisymmetry_coo
!________________________________________________________________________________________!


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
