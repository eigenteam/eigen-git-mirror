/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: frob_matrix_norm.cc,v 1.3 2003/11/30 08:26:25 opetzold Exp $
 */

#include <iostream>
#include <tvmet/Matrix.h>
#include <tvmet/xpr/Vector.h>

using namespace std;

template<class T, int Rows, int Cols>
double
frob_norm(const tvmet::Matrix<T, Rows, Cols>& M) {
  return std::sqrt(M(0,0)*M(0,0) + M(1,0)*M(1,0) + M(2,0)*M(2,0)
	  + M(0,1)*M(0,1) + M(1,1)*M(1,1) + M(2,1)*M(2,1)
	  + M(0,2)*M(0,2) + M(1,2)*M(1,2) + M(2,2)*M(2,2));
}

namespace tvmet {
  template<class T, int Rows, int Cols>
  typename NumericTraits<T>::float_type
  norm(const Matrix<T, Rows, Cols>& M) {
    return std::sqrt( sum( diag( MtM_prod(M,M) ) ) );
  }
}

int main()
{
  typedef tvmet::Matrix<double,3,3>	matrix_type;

  matrix_type M;

  M = 1,2,3,4,5,6,7,8,9;
  cout << M << endl;

  cout << "handopt norm = " << frob_norm(M) << endl;
  cout << "tvmet::norm  = " << tvmet::norm(M) << endl;
}

/*
   gcc 3.3 produce for the hand optimized frob_norm:

_Z9frob_normIdLj3ELj3EEdRKN5tvmet6MatrixIT_XT0_EXT1_EEE:
.LFB3210:
	pushl	%ebp
.LCFI6:
	movl	%esp, %ebp
.LCFI7:
	subl	$8, %esp
.LCFI8:
	movl	8(%ebp), %eax
	fldl	(%eax)
	fldl	24(%eax)
	fxch	%st(1)
	fmul	%st(0), %st
	fxch	%st(1)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	48(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	8(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	32(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	56(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	16(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	40(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fldl	64(%eax)
	fmul	%st(0), %st
	faddp	%st, %st(1)
	fld	%st(0)
	fsqrt
	fucom	%st(0)
	fnstsw	%ax
	sahf
	jp	.L189
	jne	.L189
	fstp	%st(1)
.L186:
	leave
	ret

*/

/*
   gcc 3.3 produce the norm function using tvmet 1.3.0:

_ZN5tvmet4normIdLj3ELj3EEENS_13NumericTraitsIT_E10float_typeERKNS_6MatrixIS2_XT0_EXT1_EEE:
.LFB3252:
.L194:
.L198:
.L203:
.L207:
.L212:
.L225:
.L238:
.L251:
	pushl	%ebp
.LCFI9:
	movl	%esp, %ebp
.LCFI10:
	subl	$56, %esp
.LCFI11:
	movl	8(%ebp), %edx
	leal	-24(%ebp), %eax
	movl	%eax, -12(%ebp)
	leal	-12(%ebp), %eax
	fldl	24(%edx)
	fldl	48(%edx)
	fldl	(%edx)
	fxch	%st(2)
	fmul	%st(0), %st
	fxch	%st(1)
	movl	%eax, -28(%ebp)
	fmul	%st(0), %st
	fxch	%st(2)
	movl	%edx, -24(%ebp)
	movl	%edx, -20(%ebp)
	fmul	%st(0), %st
	fldl	8(%edx)
	fxch	%st(2)
	faddp	%st, %st(3)
	fldl	56(%edx)
	fxch	%st(2)
	fmul	%st(0), %st
	fxch	%st(1)
	faddp	%st, %st(3)
	fldl	32(%edx)
	fxch	%st(2)
	fmul	%st(0), %st
	fxch	%st(2)
	fmul	%st(0), %st
	fldl	16(%edx)
	fxch	%st(1)
	faddp	%st, %st(3)
	fmul	%st(0), %st
	fldl	64(%edx)
	fxch	%st(2)
	faddp	%st, %st(3)
	fldl	40(%edx)
	fxch	%st(2)
	fmul	%st(0), %st
	fxch	%st(2)
	fmul	%st(0), %st
	faddp	%st, %st(2)
	faddp	%st, %st(1)
	faddp	%st, %st(1)
	faddp	%st, %st(1)
	fld	%st(0)
	fsqrt
	fucom	%st(0)
	fnstsw	%ax
	sahf
	jp	.L265
	jne	.L265
	fstp	%st(1)
.L261:
	fstpl	-8(%ebp)
	fldl	-8(%ebp)
	leave
	ret

*/
