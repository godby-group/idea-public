!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name: MKL parallelisation                                                          !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Author(s): Jack Wetherell                                                          !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Description:                                                                       !
! Uses MKL to parallelise the solving of Ax=b and C=AB                               !
!                                                                                    !
!                                                                                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Notes:                                                                             !
! Compile with mkl.compiler                                                          !
!                                                                                    !
!                                                                                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!============================INTEL's BLAS MULTIPLY METHOD (C)======================================
                                                                                                  
	SUBROUTINE MKL_MVMULTIPLY_C(N, A, IA, NJA, JA, NRHS, B, Y)                                
Cf2py intent(in) N, A, IA, NJA, JA, NRHS, B                                                       
Cf2py intent(out) Y                                                                               
                                                                                                  
	CHARACTER*1 transa                                                                        
	INTEGER     n, nrhs                                                                       
	INTEGER     ia(n+1)                                                                       
	INTEGER     ja(nja)                                                                       
	COMPLEX*16  a(nja)                                                                        
	COMPLEX*16  b(n)                                                                          
	COMPLEX*16  y(n)                                                                          
                                                                                                  
	transa = 'N'                                                                              
                                                                                                  
	CALL mkl_zcsrgemv(transa, n, a, ia, ja, b, y)                                             
                                                                                                  
	RETURN                                                                                    
	END                                                                                       
                                                                                                  
!==================================================================================================

!===========================INTEL's BLAS MULTIPLY METHOD (DP)======================================

	SUBROUTINE MVM_DP(N, A, IA, NJA, JA, NRHS, B, Y) 
	intent(in) N, A, IA, NJA, JA, NRHS, B
	intent(out) Y

	CHARACTER*1       transa 
	INTEGER           n, nrhs
	INTEGER           ia(n+1)
	INTEGER           ja(nja)
	DOUBLE PRECISION  a(nja)
	DOUBLE PRECISION  b(n)
	DOUBLE PRECISION  y(n)

	transa = 'N'

	CALL mkl_dcsrgemv(transa, n, a, ia, ja, b, y)

	RETURN
	END

!=================================================================================================

!==================SPLIT A VECTOR (X) INTO REAL (X1) and IM (X2) PARTS============================

	SUBROUTINE MKL_SPLIT(N, X, X1, X2) 
Cf2py intent(in) N, X
Cf2py intent(out) X1, X2

	COMPLEX*16        x(n)
	DOUBLE PRECISION  x1(n)
	DOUBLE PRECISION  x2(n)

	CALL omp_set_num_threads(omp_get_max_threads())

	!$OMP PARALLEL SHARED(x,x1,x2,n)
	!$OMP DO
	DO i = 1, n
	x1(i) = DREAL(x(i))
	x2(i) = DIMAG(x(i))  
	ENDDO
	!$OMP END DO NOWAIT
	!$OMP END PARALLEL	

	RETURN
	END

!================================================================================================

!==========================COMBINE (X1) and (X2) back into (X)===================================

	SUBROUTINE MKL_COMB(N, X1, X2, X) 
Cf2py intent(in) N, X1, X2
Cf2py intent(out) X

	COMPLEX*16        x(n)
	DOUBLE PRECISION  x1(n)
	DOUBLE PRECISION  x2(n)

	CALL omp_set_num_threads(omp_get_max_threads())

	!$OMP PARALLEL SHARED(x,x1,x2,n)
	!$OMP DO
	DO i = 1, n
	x(i) = DCMPLX(x1(i),x2(i))
	ENDDO
	!$OMP END DO NOWAIT
	!$OMP END PARALLEL	

	RETURN
	END

!================================================================================================

!===========================INTEL's ITERATIVE SOLVER=============================================

	SUBROUTINE MKL_ISOLVE(N, A, IA, NJA, JA, NRHS, B, X0, X) 
Cf2py intent(in) N, A, IA, NJA, JA, NRHS, B, X0
Cf2py intent(out) X

	INTEGER     		n, nrhs, RCI_request, itercount
	INTEGER     		ia(n+1)
	INTEGER     		ja(nja)
	INTEGER			ipar(128)
	DOUBLE PRECISION  	a(nja)
	DOUBLE PRECISION  	b(n)
	DOUBLE PRECISION  	x(n)
	DOUBLE PRECISION  	x0(n)
	DOUBLE PRECISION 	dpar(128)
	DOUBLE PRECISION,	dimension(:), allocatable:: tmp     
	
	x = x0
 	allocate(tmp((2*min(150,n)+1)*n+min(150,n)*(min(150,n)+9)/2+1))
	CALL dfgmres_init(n, x, b, RCI_request, ipar, dpar, tmp)             ! Initialise the solver   
	ipar(8) =  1                                                         ! Perform stopping tests (Maximum itarations)
	ipar(9) =  1                                                         ! Perform stopping tests (Residuals)
	ipar(10) = 0                                                         ! No user defined stopping tests
	dpar(1) = 1.0D-11                                                    ! Relative tollerance                                   

	CALL dfgmres_check(n, x, b, RCI_request, ipar, dpar, tmp)            ! Check that the parameters are not contradictory
	CALL dfgmres(n, x, b, RCI_request, ipar, dpar, tmp)                  ! Run the algorithm (one iteration)
	CALL mvm_dp(n, a, ia, nja, ja, nrhs, tmp(ipar(22)), tmp(ipar(23)))   ! Multipy the resultant vector by A
	CALL dfgmres(n, x, b, RCI_request, ipar, dpar, tmp)                  ! Run the algorithm (one iteration)

	DO WHILE(dpar(5) > 1.0D-11)                                          ! Run algorithm until absolute residual is below 10^-11
	CALL dfgmres(n, x, b, RCI_request, ipar, dpar, tmp)                  ! Run the algorithm (one iteration) 
	CALL mvm_dp(n, a, ia, nja, ja, nrhs, tmp(ipar(22)), tmp(ipar(23)))   ! Multipy the resultant vector by A
	CALL dfgmres(n, x, b, RCI_request, ipar, dpar, tmp)                  ! Run the algorithm (one iteration)
	ENDDO

	CALL dfgmres_get(n, x, b, RCI_request, ipar, dpar, tmp, itercount)   ! Get the final value of x

	RETURN
	END

!================================================================================================


