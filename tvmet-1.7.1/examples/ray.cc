#include <iostream>
#include <cmath>

#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

typedef Vector<double,3>	vector3d;

void reflect(vector3d& reflection, const vector3d& ray, const vector3d& surfaceNormal)
{
  // The surface normal must be unit length to use this equation.
  reflection = ray - 2 * dot(ray,surfaceNormal) * surfaceNormal;

  // expression printing
  //  cout << (ray - 2 * dot(ray,surfaceNormal) * surfaceNormal) << endl;
}

int main()
{
  vector3d x, y, z;

  // y will be the incident ray
  y[0] = 1;
  y[1] = 0;
  y[2] = -1;

  // z is the surface normal
  z[0] = 0;
  z[1] = 0;
  z[2] = 1;

  reflect(x, y, z);

  cout << "Reflected ray is: [ " << x[0] << " " << x[1] << " " << x[2]
       << " ]" << endl;
}

/*****************************************************************************************
gcc 3.2.0 produce this code (i586) using tvmet 1.3.0:

_Z7reflectRN5tvmet6VectorIdLj3EEERKS1_S4_:
.LFB2757:
.L8:
.L18:
.L22:
.L28:
.L32:
.L38:
.L44:
.L48:
	pushl	%ebp
.LCFI0:
	movl	%esp, %ebp
.LCFI1:
	leal	-32(%ebp), %eax
	pushl	%ebx
.LCFI2:
	subl	$52, %esp
.LCFI3:
	movl	16(%ebp), %edx
	movl	%eax, -24(%ebp)
	movl	12(%ebp), %ecx
	leal	-36(%ebp), %eax
	movl	%eax, -20(%ebp)
	movl	8(%ebp), %ebx
	leal	-24(%ebp), %eax
	fldl	8(%edx)
	fldl	16(%edx)
	fmull	16(%ecx)
	fxch	%st(1)
	movl	%eax, -12(%ebp)
	leal	-52(%ebp), %eax
	fmull	8(%ecx)
	movl	%eax, -48(%ebp)
	leal	-12(%ebp), %eax
	fldl	(%edx)
	fmull	(%ecx)
	fxch	%st(1)
	movl	%eax, -44(%ebp)
	faddp	%st, %st(2)
	faddp	%st, %st(1)
	fadd	%st(0), %st
	fld	%st(0)
	fstl	-32(%ebp)
	fxch	%st(1)
	fmull	(%edx)
	fsubrl	(%ecx)
	fstpl	(%ebx)
	fld	%st(0)
	fmull	8(%edx)
	fsubrl	8(%ecx)
	fstpl	8(%ebx)
	fmull	16(%edx)
	fsubrl	16(%ecx)
	fstpl	16(%ebx)
	addl	$52, %esp
	popl	%ebx
	popl	%ebp
	ret

*****************************************************************************************/
