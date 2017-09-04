! FILE: ideafort.f90
!
!	xygrid() constructs the mapping between the 1D wavefunction vector index
!	and the electron coordinates it indexes
!
        SUBROUTINE xygrid(N_e,N_x,N,ie,xy)
        IMPLICIT NONE
!       Inputs
!	N_e = the number of electrons
!	N_x = the number of grid points in the simulation region
!	N = N_x**N_e
!	ie = electron index
        INTEGER(KIND=4), INTENT(IN) :: N_e, N_x, N, ie
!       Outputs
!	xy maps a 1D wavefunction vector index onto an electron coordinate for a given electron
        INTEGER(KIND=4), INTENT(OUT) :: xy(N)
!       Local variables
        INTEGER(KIND=4) :: ix, WHEN_TO_INCREMENT, WHEN_TO_RESET, counter, inc, res
!
        xy(:) = 0
!
        WHEN_TO_INCREMENT = N_x**(N_e-ie)
        WHEN_TO_RESET = N_x**(N_e-ie+1)
        counter = 0
!
!	Iterate through the indices of the 1D wavefunction vector
!	Each index ix corresponds to N_e electron coordinates
        DO ix=1,N
!
                xy(ix) = counter
!
!		Check if it is time to increment the counter
                inc = MOD(ix,WHEN_TO_INCREMENT)
                IF (inc == 0) THEN
                        counter = counter + 1
                END IF
!
!		Check if it is time to reset the counter
                res = MOD(ix,WHEN_TO_RESET)
                IF (res == 0) THEN
                        counter = 0
                END IF
!
        END DO
        END
!
!
!	As xygrid, but maps the individual electron coordinate indices to the reduced wavefunction vector index
        SUBROUTINE downgrid(N_e, N, xy, sgrid)
        IMPLICIT NONE
!       Inputs
!	N_e = number of electrons
!	N = total number of wavefunction coordinates
!	xy = mapping between individual electron coordinates and full wavefunction vector
        INTEGER(KIND=4), INTENT(IN) :: N_e, N
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e)
!       Outputs
!	sgrid = mapping between individual electron coordinates and reduced wavefunction vector
        INTEGER(KIND=4), INTENT(OUT) :: sgrid(N)
!       Local variables
!	ind = individual electron coordinate indices
        INTEGER(KIND=4) :: i1, i2, counter, i, process
        INTEGER(KIND=4) :: ind(N_e)
!
        counter = 0
        sgrid(:) = 0
!
!	Iterate through each index of the full wavefunction vector
        DO i=1,N
!
!	    Get the electron coordinate indices for this wavefunction index
            ind = xy(i,:)
!
!	    Check if this coordinate is present in the reduced wavefunction
            process = 1
            DO i1=1,N_e-1
                DO i2=i1+1,N_e
                    IF ( ind(i2) .GE. ind(i1) ) THEN
                        process = 0
                    END IF
                END DO
            END DO
!
!	    Map the reduced wavefunction index to these electron coordinates
            IF ( process == 1 ) THEN
                sgrid(i) = counter
                counter = counter + 1
            END IF
!
        END DO
!
        END
!
!
!	Parallelised subroutine:
!	Constructs the reduction and expansion matrices
        SUBROUTINE antisym(N_e, Nxs, N, N_x, N_perms, xy, sgrid, downrows, downcols, downvals, uprows, upcols, upvals)
        USE omp_lib
        IMPLICIT NONE
!       Inputs
!	N_e = number of electrons
!	Nxs = number of elements in reduced wavefunction
!	N = number of elements in full wavefunction
!	N_x = number of grid points in simulation region
!	N_perms = number of unique permutations of each set of electron coordinates
        INTEGER(KIND=4), INTENT(IN) :: N_e, Nxs, N, N_x, N_perms
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e), sgrid(N)
!       Outputs
!	downrows = the row indices of the nonzero elements of the reduction matrix
!	downcols = the column indices of the nonzero elements of the reduction matrix
!	uprows = the row indices of the nonzero elements of the expansion matrix
!	upcols = the column indices of the nonzero elements of the expansion matrix
        INTEGER(KIND=4), INTENT(OUT) :: downrows(Nxs), downcols(Nxs), uprows(N), upcols(N)
!	downvals = the nonzero elements of the reduction matrix
!	upvals = the nonzero elements of the expansion matrix
        REAL*8, INTENT(OUT) :: downvals(Nxs), upvals(N)
!       Local variables
!	perms = all permutations of the electron coordinates
!	swaps = the number of times two indices of original coordinates must be swapped to reach current permutation
        INTEGER(KIND=4) :: i, process, iperm, swapper, ie, je, j
        INTEGER(KIND=4) :: perms(N_perms,N_e), swaps(N_perms)
        REAL*8 :: exchange
!
!	Initialise arrays
        downrows(:) = 0
        downcols(:) = 0
        downvals(:) = 0.0d+0
        uprows(:) = 0
        upcols(:) = 0
        upvals(:) = 0.0d+0
        perms(:,:) = 0
        swaps(:) = 0
!
!	Initialise OpenMP
        CALL omp_set_num_threads( omp_get_max_threads() )
!
!	Parallelise over (full) wavefunction vector index
      !$OMP PARALLEL PRIVATE(process,perms,swaps,iperm,j) &
      !$OMP & SHARED(N,xy,downrows,sgrid,downcols,downvals,N_perms,N_e,uprows,upcols,upvals)
      !$OMP DO
        DO i=1,N
!
!	    Check to see if current index is present in reduced wavefunction
            process = 1
            DO ie=1,N_e-1
                 IF ( xy(i,ie+1) .GE. xy(i,ie) ) THEN
                     process = 0
                 END IF
            END DO
!
            IF (process == 1) THEN
!
!               Calculate the C_down indices and values
!
                downrows(sgrid(i)) = sgrid(i)-1
                downcols(sgrid(i)) = i-1
                downvals(sgrid(i)) = 1.0
!
!               Calculate the C_up indices and values
!
!		Get all permutations of this set of electron coordinates
                CALL permutations(xy(i,:), N_perms, N_e, perms, swaps)
!
!		Get the full wavefunction index for each permutation
                DO iperm=1,N_perms
                     CALL get_index(N_e, N_x, perms(iperm,:), j)
                     uprows(j) = j - 1
                     upcols(j) = sgrid(i) - 1
!		     Calculate the exchange eigenvalue
                     upvals(j) = (-1.0d+0)**swaps(iperm)
                END DO
!
            END IF
!
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
!	Given a set of electron coordinates, returns all unique permutations of that set
        SUBROUTINE permutations(ind, N_perms, N_e, perms, swaps)
        IMPLICIT NONE
!        Input variables
        INTEGER(KIND=4), INTENT(IN) :: ind(N_e), N_perms, N_e
!        Output variables
        INTEGER(KIND=4), INTENT(OUT) :: swaps(N_perms), perms(N_perms, N_e)
!        Local variables
        INTEGER(KIND=4) :: ie, counter, swapcount, lind(N_e)
!
        swaps(:) = 0
        perms(:,:) = 0
        DO ie=1,N_e
            lind(ie) = ind(ie)
        END DO
!
        counter = 1
        swapcount = 0
        CALL one_permutation(lind, N_e, counter, N_e, N_perms, perms, swaps, swapcount)
!
        END
!
!
!	Calculates next permutation of index
!	For use by permutations() subroutine only
        RECURSIVE SUBROUTINE one_permutation(ind, n, counter, N_e, N_perms, perms, swaps, swapcount)
        IMPLICIT NONE
!        Input variables
        INTEGER(KIND=4), INTENT(IN) :: n, N_e, N_perms
!        Input/output variables
        INTEGER(KIND=4), INTENT(INOUT) :: perms(N_perms,N_e), counter, swapcount, &
        & swaps(N_perms), ind(N_e)
!        Output variables
!        Local variables
        INTEGER(KIND=4) :: ie, i
!
        IF (n == 1) THEN
            DO ie=1,N_e
                perms(counter,ie) = ind(ie)
            END DO
            swaps(counter) = swapcount
            counter = counter + 1
        ELSE
            DO i=1,n
                CALL one_permutation(ind, n-1, counter, N_e, N_perms, perms, swaps, swapcount)
                IF (MOD(n,2) == 0) THEN
                    CALL swap(N_e, ind, 1, n, swapcount)
                ELSE
                    CALL swap(N_e, ind, i, n, swapcount)
                END IF
            END DO
        END IF
!
        END
!
!
!	Swaps the x^th and y^th elements of a set of coordinates
!	For use by permutations() subroutine only
        SUBROUTINE swap(N_e, ind, x, y, swapcount)
        IMPLICIT NONE
!       Input variables
        INTEGER(KIND=4), INTENT(IN) :: N_e, x, y
!       In/output variables
        INTEGER(KIND=4), INTENT(INOUT) :: swapcount, ind(N_e)
!       Local variables
        INTEGER(KIND=4) :: t
!
        t = ind(x)
        ind(x) = ind(y)
        ind(y) = t
        IF ( x .ne. y  ) THEN
            swapcount = swapcount + 1
        END IF
!
        END
!
!
!	Calculates the full wavefunction index for a given set of electron coordinates
        SUBROUTINE get_index(N_e, N_x, ind, j)
        IMPLICIT NONE
        INTEGER(KIND=4), INTENT(IN) :: N_e, N_x
        INTEGER(KIND=4), INTENT(IN) :: ind(N_e)
        INTEGER(KIND=4), INTENT(OUT) :: j
        INTEGER(KIND=4) :: ie
!
        j = 1
        DO ie=1,N_e
            j = j + (ind(ie)-1)*(N_x**(N_e-ie))
        END DO
!
        END
!
!
!	Constructs the diagonal of the Coulomb operator matrix
        SUBROUTINE coulomb(N_e, N, N_x, N_cells, xy, dx, C, w)
        USE omp_lib
        IMPLICIT NONE
!       Inputs
        INTEGER(KIND=4), INTENT(IN) :: N_e, N, N_x, N_cells
        REAL*8, INTENT(IN) :: dx, C
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e)
!       Outputs
        REAL*8, INTENT(OUT) :: w(N)
!       Locals
        INTEGER(KIND=4) :: i, ie, je, j
!
!	Initialise the array
        w(:) = 0.0d+0
!
!	Initialise shared memory parallelisation
        CALL omp_set_num_threads( omp_get_max_threads() )
!
!	Parallelise over index of diagonal
      !$OMP PARALLEL PRIVATE(ie,je,j) &
      !$OMP & SHARED(N,N_e,w,xy,dx,C,N_cells)
      !$OMP DO
        DO i=1,N
!
!	    For each diagonal, iterative over each pair of electrons
            DO ie=1,N_e
                DO je=1,ie-1
!
!		    Calculate the intracell interactions
                    w(i) = w(i) + 1.0/( ABS(xy(i,ie)-xy(i,je))*dx + C )
!
!		    Calculate the intercell interactions
                    DO j=2,N_cells
                        w(i) = w(i) + 1.0/(  ABS(xy(i,ie)-xy(i,je)+(j-1)*N_x)*dx + C  )
                        w(i) = w(i) + 1.0/(  ABS(xy(i,ie)-xy(i,je)-(j-1)*N_x)*dx + C  )
                    END DO
!
                END DO
            END DO
!
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
!	Calculates the Hartree potential for a 1D homogenous electron gas of density N_e/L
        SUBROUTINE hartree(N_e, N_x, N, N_cells, dx, L, C, xy, inc_primary, v)
        USE omp_lib
        IMPLICIT NONE
!       Inputs
        INTEGER(KIND=4), INTENT(IN) :: N_e, N_x, N, N_cells, inc_primary
        REAL*8, INTENT(IN) :: dx, L, C
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e)
!       Outputs
        REAL*8, INTENT(OUT) :: v(N)
!       Locals
        INTEGER(KIND=4) :: i1, i2, i3
        REAL*8 :: v_p(N_x)
!
        v(:) = 0.0d+0
        v_p(:) = 0.0d+0
!
        CALL omp_set_num_threads( omp_get_max_threads() )
!        CALL omp_set_num_threads( 1 )
      !$OMP PARALLEL PRIVATE(i2,i3) &
      !$OMP & SHARED(N_x,N_cells,v_p,N_e,dx,L,C)
      !$OMP DO
        DO i1=1,N_x
            DO i2=1,(2*N_cells+1)*N_x
                i3 = i2 - N_x*N_cells
                v_p(i1) = v_p(i1) + (N_e/L)*dx/(ABS(i1-i3)*dx + C)
            END DO
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
      !$OMP PARALLEL PRIVATE(i2) &
      !$OMP & SHARED(N,N_e,v,v_p,xy)
      !$OMP DO
        DO i1=1,N
            DO i2=1,N_e
                v(i1) = v(i1) + v_p(xy(i1,i2))
            END DO
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
!	Calculates the row and column indices and values of nonzero elements in the many-electron kinetic energy matrix
!	Includes finite, periodic or twisted boundary conditions
        SUBROUTINE kinetic(N_e, N, N_fdc, N_x, N_k, offsets, xy, fdc, phase, trows, tcols, tvals)
        USE omp_lib
        IMPLICIT NONE
!	Input variables
!	N_e = number of electrons in system
!	N = number of full wavefunction elements
!	N_fdc = number of finite difference coefficients in second derivative
!	N_x = number of grid points in simulation region
!	phase --> exp(i*phase*pi) = phase difference in wavefunction at either end of simulation region
!	N_k --> N_k*N = number of nonzero kinetic energy matrix elements
!	offsets = the offsets from diagonal of each finite difference coefficient in single-particle kinetic energy matrix
!	fdc = finite difference coefficients of second derivative
        INTEGER(KIND=4), INTENT(IN) :: N_e, N, N_fdc, N_x
        REAL*8, INTENT(IN) :: phase
        INTEGER*8, INTENT(IN) :: N_k
        INTEGER(KIND=4), INTENT(IN) :: offsets(N_fdc), xy(N,N_e)
        REAL*8, INTENT(IN) :: fdc(N_fdc)
!	Output variables
!	trows = row indices of nonzero elements of many-body kinetic energy matrix
!	tcols = column indices of nonzero elements of many-body kinetic energy matrix
!	tvals = values of nonzero elements of many-body kinetic energy matrix
        INTEGER(KIND=4), INTENT(OUT) :: trows(N_k*N), tcols(N_k*N)
        COMPLEX*16, INTENT(OUT) :: tvals(N_k*N)
!	Local variables
        INTEGER(KIND=4) :: i, j, ie, ik, je, ind, midfd
        INTEGER(KIND=4) :: xind(N_e), permind(N_e), zind(N_e), yind(N_e)
        INTEGER*8 :: counter, kindex
        REAL*8, PARAMETER :: pi = 3.1415927
        COMPLEX*16 :: t, aL, aR
!
        CALL omp_set_num_threads( omp_get_max_threads() )
        counter = 0
        midfd = (N_fdc+1)/2
        DO i=1,N_e
            trows(i) = 0
            tcols(i) = 0
            tvals(i) = CMPLX(0.0d+0,0.0d+0)
        END DO
!
        IF (phase == 0.0) THEN
            aL = CMPLX(0.0,0.0)
            aR = CMPLX(0.0,0.0)
        ELSE
            aL = CMPLX(cos(phase*pi),-sin(phase*pi))
            aR = CMPLX(cos(phase*pi),sin(phase*pi))
        END IF
!
      !$OMP PARALLEL PRIVATE(kindex, xind, counter, ie, ik, yind, t, j) &
      !$OMP & SHARED(N, N_x, N_fdc, trows, tcols, tvals, fdc, midfd, xy, offsets, aR, aL)
      !$OMP DO
        DO i=1,N
!
            kindex = (i-1)*((N_e*(N_fdc-1))+1) + 1
            trows(kindex) = i - 1
            tcols(kindex) = i - 1
            tvals(kindex) = N_e*fdc(midfd)
!
            xind = xy(i,:)
            counter = 0
!
            DO ie=1,N_e
!
                DO ik=1,N_fdc
!
                    IF ( ik .NE. midfd ) THEN
!
                        yind = xind
                        yind(ie) = yind(ie) + offsets(ik)
                        t = CMPLX(fdc(ik),0.0)
                        IF ( yind(ie) .LE. 0 ) THEN
                            yind(ie) = yind(ie) + N_x
                            t = t*aR
                        ELSE IF ( yind(ie) > N_x ) THEN
                            yind(ie) = yind(ie) - N_x
                            t = t*aL
                        END IF
!
                        CALL get_index(N_e, N_x, yind, j)
                        counter = counter + 1
                        kindex = (i-1)*((N_e*(N_fdc-1))+1) + 1 + counter
                        trows(kindex) = i - 1
                        tcols(kindex) = j - 1
                        tvals(kindex) = t
!
                    END IF
                END DO
            END DO
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
        SUBROUTINE density(N_cd, N_x, N, N_e, xy, dx, phi, rho)
        IMPLICIT NONE
!       Inputs
        INTEGER(KIND=4), INTENT(IN) :: N_cd, N_x, N, N_e
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e)
        REAL*8, INTENT(IN) :: dx
        COMPLEX*16, INTENT(IN) :: phi(N_cd)
!       Outputs
        REAL*8, INTENT(OUT) :: rho(N_x)
!       Locals
        INTEGER(KIND=4) :: i, rho_x, j, ie
        INTEGER(KIND=4) :: ind(N_e), yind(N_e), sgrid(N)
        REAL*8 :: dn
!
        rho(:) = 0
!
        CALL downgrid_fortran(N_e, N, xy, sgrid)
!
      !$OMP PARALLEL PRIVATE(ie,ind,yind,rho_x,j,dn) &
      !$OMP & SHARED(N,N_e,xy,N_x,sgrid,phi,rho)
      !$OMP DO
        DO i=1,N
!
            DO ie=1,N_e
                ind(ie) = xy(i,ie)
            END DO
            rho_x = ind(1)
            CALL reverse_sort(N_e, ind, yind)
            CALL get_index(N_e, N_x, yind, j)
!
            IF ( sgrid(j) .NE. 0 ) THEN
                dn = N_e*(REAL(phi(sgrid(j)))**2 + AIMAG(phi(sgrid(j)))**2)
              !$OMP CRITICAL
                rho(rho_x) = rho(rho_x) + (dx**(N_e-1))*dn
              !$OMP END CRITICAL
            END IF
!
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
        SUBROUTINE reverse_sort(N_e, ind, sort)
        IMPLICIT NONE
!       Input variables
        INTEGER(KIND=4), INTENT(IN) :: N_e
        INTEGER(KIND=4), INTENT(IN) :: ind(N_e)
!       Output variables
        INTEGER(KIND=4), INTENT(OUT) :: sort(N_e)
!       Local variables
        INTEGER(KIND=4) :: maxi, ie, mind, je, process, done(N_e), maxie
!
        mind = 1
        sort(:) = 0
        done(:) = 0
!
        DO
!
            maxi = 0
            DO ie=1,N_e
                IF ( ind(ie) >= maxi ) THEN
                    process = 1
                    DO je=1,N_e
                        IF ( done(je) == ie ) THEN
                            process = 0
                        END IF
                    END DO
                    IF ( process == 1) THEN
                        maxi = ind(ie)
                        maxie = ie
                    END IF
                END IF
            END DO
            sort(mind) = maxi
            done(mind) = maxie
            mind = mind + 1
!
            IF ( mind == N_e+1 ) THEN
                EXIT
            END IF
!
        END DO
!
        END
!
!
        SUBROUTINE downgrid_fortran(N_e, N, xy, sgrid)
        IMPLICIT NONE
!        Inputs
        INTEGER(KIND=4), INTENT(IN) :: N_e, N
        INTEGER(KIND=4), INTENT(IN) :: xy(N,N_e)
!        Outputs
        INTEGER(KIND=4), INTENT(OUT) :: sgrid(N)
!        Local variables
        INTEGER(KIND=4) :: i1, i2, counter, i, process
        INTEGER(KIND=4) :: ind(N_e)
!
        sgrid(:) = 0
!
        counter = 0
        DO i=1,N
            ind = xy(i,:)
            process = 1
            DO i1=1,N_e-1
                DO i2=i1+1,N_e
                    IF ( ind(i2) .GE. ind(i1) ) THEN
                        process = 0
                    END IF
                END DO
            END DO
            IF ( process == 1 ) THEN
                counter = counter + 1
                sgrid(i) = counter
            END IF
        END DO
!
        END
!
!
        SUBROUTINE slater_determinant(N_e, N_perms, N, N_x, xy, u, phi)
        USE omp_lib
        IMPLICIT NONE
!       Input variables
        INTEGER*4, INTENT(IN) :: N_e, N_perms, N, N_x
        INTEGER*4, INTENT(IN) :: xy(N,N_e)
        COMPLEX*16, INTENT(IN) :: u(N_x,N_e)
!       Output variables
        COMPLEX*16, INTENT(OUT) :: phi(N)
!       Local variables
        INTEGER*4 :: ie, i, iperm
        INTEGER*4, DIMENSION(:), ALLOCATABLE :: ind, swaps, perm
        INTEGER*4, DIMENSION(:,:), ALLOCATABLE :: perms
        COMPLEX*16 :: psi
!
        CALL omp_set_num_threads( omp_get_max_threads() )
!        CALL omp_set_num_threads( 1 )
!
!       Initialisation
        ALLOCATE(ind(N_e))
        ALLOCATE(swaps(N_perms))
        ALLOCATE(perm(N_e))
        ALLOCATE(perms(N_perms,N_e))
        DO ie=1,N_e
            ind(ie) = ie
        END DO
        swaps(:) = 0
        perms(:,:) = 0
        phi(:) = CMPLX(0d+0,0d+0)
!
        CALL permutations(ind, N_perms, N_e, perms, swaps)
!
      !$OMP PARALLEL PRIVATE(ie,ind,iperm,perm,psi) &
      !$OMP & SHARED(N,N_e,xy,N_perms,perms,N_x,swaps,u,phi)
      !$OMP DO
        DO i=1,N
            DO ie=1,N_e
                ind(ie) = xy(i,ie)
            END DO
            DO iperm=1,N_perms
                DO ie=1,N_e
                    perm(ie) = perms(iperm,ie)
                END DO
                CALL product_wavefunction(N_e, N_x, N_perms, swaps(iperm), u, ind, perm, psi)
                phi(i) = phi(i) + psi
            END DO
        END DO
      !$OMP END DO
      !$OMP END PARALLEL
!
        END
!
!
        SUBROUTINE product_wavefunction(N_e, N_x, N_perms, swap, u, ind, perm, psi)
        IMPLICIT NONE
!       Input variables
        INTEGER*4, INTENT(IN) :: N_e, N_x, N_perms, swap
        INTEGER*4, INTENT(IN) :: ind(N_e), perm(N_e)
        COMPLEX*16, INTENT(IN) :: u(N_x,N_e)
!       Output variables
        COMPLEX*16, INTENT(OUT) :: psi
!       Local variables
        INTEGER*4 :: ie
!
        psi = (-1.0)**swap
        DO ie=1,N_e
            psi = psi*u(ind(ie),perm(ie))
        END DO
        psi = psi/SQRT(1.0d+0*N_perms)
!
        END
!
!
! END FILE ideafort.f90
