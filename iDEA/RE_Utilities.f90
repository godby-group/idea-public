!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name: RE parallelisation                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Author(s): Ed Higgins, Aaron Hopkinson, Matthew Smith                              !
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

!* f2py -c --fcompiler=intelem -L${MKLROOT}/lib/intel64/ -lmkl_rt -m RE_Utilities RE_Utilities.f --f90flags='-openmp' -lgomp --opt='-fast'

  subroutine continuity_eqn(cd,nx,dx,dt,n_new,n_old)
  implicit none
!f2py intent(out) cd
!f2py intent(in) nx, dx, dt, n_new, n_old
!f2py depend(nx) cd, n_new, n_old

  double precision, dimension(nx),  intent(out) ::  cd
  integer,                          intent(in)  ::  nx
  double precision,                 intent(in)  ::  dx
  double precision,                 intent(in)  ::  dt
  double precision, dimension(nx),  intent(in)  ::  n_new
  double precision, dimension(nx),  intent(in)  ::  n_old
  integer           ::  i, k  
  double precision  :: prefac

  prefac = dx/dt
  cd(:) = 0.0d0
  do i=1,nx
    do k=1, i
      cd(i) = cd(i) - prefac*(n_new(k)-n_old(k))
    end do
  end do

  end subroutine continuity_eqn


  subroutine compare(j_check,lenj,j,j0,tol)
  implicit none
!f2py intent(out) j_check
!f2py intent(in)  lenj,j,j0,tol
!f2py depend(lenj) j, j0

  logical,                            intent(out) ::  j_check
  integer,                            intent(in)  ::  lenj
  double precision, dimension(lenj),  intent(in)  ::  j
  double precision, dimension(lenj),  intent(in)  ::  j0
  double precision,                   intent(in)  ::  tol
  
  j_check = all( abs(j-j0) < tol )

  end subroutine compare





