!========================================================================================!
! CONSTRUCT THE SYSTEMS HAMILTONIAN OPERATOR IN COOrdinate FORM (3 electrons).           !
!========================================================================================!
!
!This module contains one subroutine and one function:
!
!    construct_hamiltonian_coo (int,real): Populates three input arrays with a
!        COOrdinate form of the systems Hamiltonian. These can then be constructed into
!        a COOrdinate sparse matrix.
!
!    single_index (int): Creates a single unique index for all permutations of the three
!        electron indices.
!
!    potential (real): Calculates the potential contribution to a particular element along 
!        the main diagonal of the Hamiltonian matrix.
!
!----------------------------------------------------------------------------------------!
!Created by Mike Entwistle (me624@york.ac.uk)                                            !
!________________________________________________________________________________________!


!========================================================================================!
! CONSTRUCT HAMILTONIAN OPERATOR IN SPARSE COOrdinate FORM (3 electrons)                 !
!----------------------------------------------------------------------------------------!
!
!Populates three input arrays with a COOrdinate form of the system's Hamiltonian.
!
!    Take in 3 holding arrays and one array containing the pre-calculated band elements
!    and constructs the system's Hamiltonian. Coordinates and data are then passed back 
!    into the holding arrays. These can then be converted into a COOrdinate sparse matrix 
!    (efficient storage).
!
!    NB: As we iterate through each non-zero element, we increase our i-th element 
!        counter (i) corresponding to the total number of non-zero elements (~max_size).
!
!    Args:
!        coo_1,coo_2 (int, shape=max_size): Coordinate holding arrays.
!        coo_data (real, shape=max_size): Data for a given matrix element.
!        grid (int): Spatial grid points for the system.
!        const (real): Constant that appears in the main diagonal elements.
!        v_ext (real, shape=grid): External potential
!        v_coulomb (real, shape=grid): Coulomb potential
!        interaction_strength (real): Strength of the Coulomb interaction.
!        band_elements (real, shape=bandwidth): Band elements. 
!        bandwidth (int, optional): Number of single-particle bands.
!        max_size (int, optional): Estimate for the total number of non-zero elements in
!        the Hamiltonian.
!
!    Returns:
!        coo_1,coo_2 (int, shape=max_size): Populated coordinate arrays.
!        coo_data (real, shape=max_size): Populated data array.
!
!========================================================================================!
subroutine construct_hamiltonian_coo(coo_1, coo_2, coo_data, grid, const, v_ext,&
                                     & v_coulomb, interaction_strength, band_elements,& 
                                     & bandwidth, max_size) 

  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer            :: i, j, k, l, jkl, band

  !======================================================================================!
  ! F2PY SIGNATURE CREATION                                                              !
  !--------------------------------------------------------------------------------------!
  !f2py intent(in) :: grid, max_sizee, bandwidth
  integer :: grid, max_size, bandwidth

  !f2py intent(in, out) :: coo_1
  !f2py integer,intent(hide),depend(coo_1) :: max_size=shape(coo_1, 0)
  integer :: coo_1(0:max_size-1)

  !f2py intent(in, out) :: coo_2
  integer :: coo_2(0:max_size-1)

  !f2py intent(in) :: interaction_strength
  real (kind=dp) :: interaction_strength
  
  !f2py intent(in) :: v_ext, v_coulomb  
  real (kind=dp) :: v_ext(0:grid-1), v_coulomb(0:grid-1)

  !f2py intent(in) :: const
  real (kind=dp) :: const

  !f2py intent(in, out) :: coo_data
  real (kind=dp) :: coo_data(0:max_size-1)

  !f2py intent(in) :: band_elements
  !f2py integer,intent(hide),depend(band_elements) :: bandwidth=shape(band_elements, 0)
  real (kind=dp) :: band_elements(0:bandwidth-1)
  !______________________________________________________________________________________!


  !=======================================================!
  ! EXTERNAL FORTRAN FUNCTION:                            !
  !-------------------------------------------------------!
  integer, external :: single_index                       !
  real (kind=dp), external :: potential                   !
  !_______________________________________________________!

  !=======================================================!
  ! MAIN PROGRAM LOOP                                     !
  !=======================================================!

  ! Counter for ith element of the COO arrays.
  i = 0

  do j = bandwidth-1, grid-bandwidth
     do k = bandwidth-1, grid-bandwidth
        do l = bandwidth-1, grid-bandwidth

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
            do band = 1, bandwidth-1
                coo_1(i) = single_index(j+band, k, l, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1

                coo_1(i) = single_index(j-band, k, l, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1

                coo_1(i) = single_index(j, k+band, l, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1

                coo_1(i) = single_index(j, k-band, l, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1

                coo_1(i) = single_index(j, k, l+band, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1

                coo_1(i) = single_index(j, k, l-band, grid)
                coo_2(i) = jkl
                coo_data(i) = band_elements(band)
                i = i+1
            end do 
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do j = 0, bandwidth-2
     do k = 0, grid-1
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do j = grid-bandwidth+1, grid-1
     do k = 0, grid-1
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do k = 0, bandwidth-2
     do j = bandwidth-1, grid-bandwidth
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do k = grid-bandwidth+1, grid-1
     do j = bandwidth-1, grid-bandwidth
        do l = 0, grid-1

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do l = 0, bandwidth-2
     do j = bandwidth-1, grid-bandwidth
        do k = bandwidth-1, grid-bandwidth

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

  do l = grid-bandwidth+1, grid-1
     do j = bandwidth-1, grid-bandwidth
        do k = bandwidth-1, grid-bandwidth

           !=======================================================!
           ! ASSIGN DIAGONAL                                       !
           !=======================================================!
           jkl = single_index(j, k, l, grid)
           coo_1(i) = jkl
           coo_2(i) = jkl
           coo_data(i) = band_elements(0) + const*potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid)
           i = i+1
           !_______________________________________________________!

           !=======================================================!
           ! ASSIGN ADDITIONAL BANDS                               !
           !=======================================================!
           do band = 1, bandwidth-1
               if (j < grid-band) then
                   coo_1(i) = single_index(j+band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (j > band-1) then
                   coo_1(i) = single_index(j-band, k, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
 
               if (k < grid-band) then
                   coo_1(i) = single_index(j, k+band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (k > band-1) then
                   coo_1(i) = single_index(j, k-band, l, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l < grid-band) then
                   coo_1(i) = single_index(j, k, l+band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 

               if (l > band-1) then
                   coo_1(i) = single_index(j, k, l-band, grid)
                   coo_2(i) = jkl
                   coo_data(i) = band_elements(band)
                   i = i+1
               end if 
           end do  
           !_______________________________________________________!

        end do
     end do
  end do
  !_______________________________________________________!

end subroutine construct_hamiltonian_coo
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


!========================================================================================!
! POTENTIAL RETURN (RETURN: REAL)                                                        !
!----------------------------------------------------------------------------------------!
! Calculates the potential contribution to a particular element along the main diagonal  !
! of the Hamiltonian matrix.                                                             ! 
!========================================================================================!
function potential(j, k, l, interaction_strength, v_ext, v_coulomb, grid) result(element)

    implicit none

    integer, parameter :: dp = selected_real_kind(15, 300)
    integer, intent(in) :: j, k, l, grid
    real (kind=dp), intent(in) :: interaction_strength
    real (kind=dp), intent(in) :: v_ext(0:grid-1), v_coulomb(0:grid-1)
    real (kind=dp) :: element

    element = v_ext(j) + v_ext(k) + v_ext(l) + interaction_strength*(v_coulomb(ABS(j-k))& 
            & + v_coulomb(ABS(j-l)) + v_coulomb(ABS(k-l)))

    return

end function potential
!________________________________________________________________________________________!
