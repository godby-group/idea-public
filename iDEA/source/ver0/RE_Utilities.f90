!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name: RE parallelisation                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Author(s): Matthew Smith                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Description:                                                                       !
! Uses MKL to parallelise some functions in RE                                       !
!                                                                                    !
!                                                                                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Notes:                                                                             !
! Compile with *                                                                     !
!                                                                                    !
!                                                                                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!* f2py -c --fcompiler=intelem -L${MKLROOT}/lib/intel64/ -lmkl_rt -m RE_Utilities RE_Utilities.f90 --f90flags='-openmp' -lgomp --opt='-fast'

	SUBROUTINE continuity_eqn(j,nx,dx,dt,n,lnrow,lncol,cd,lcd) 
    
Cf2py intent(in) J, NX, DX, DT, N, LNROW, LNCOL, LCD
Cf2py intent(inout) CD

	INTEGER				j, nx, lnrow, lncol, lcd, i, k	
        INTEGER,			parameter :: dp = SELECTED_REAL_KIND(15,307)
	REAL(KIND=dp)			n(lnrow,lncol)
	REAL(KIND=dp)			cd(lcd)
	REAL(KIND=dp)			dx, dt, prefac

	prefac = dx/dt
	do i=1,nx
		do k=1, i
			cd(i) = cd(i) - prefac*(n(j,k)-n(j-1,k))
		end do
	end do

	RETURN
	END

	SUBROUTINE COMPARE(J,LENJ,J0,LENJ0,TOL,J_CHECK) 
    
Cf2py intent(in) LENJ,J0,LENJ0,TOL
Cf2py intent(out) J_CHECK

	INTEGER lenJ, lengthJ0
	DOUBLE PRECISION J(lenJ)
	DOUBLE PRECISION J0(lenJ0)
	DOUBLE PRECISION tol
	LOGICAL	J_CHECK
	
	J_CHECK = all( abs(J-J0) < tol )

	RETURN
	END





