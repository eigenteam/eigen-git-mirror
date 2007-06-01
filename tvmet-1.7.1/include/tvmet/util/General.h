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
 * $Id: General.h,v 1.9 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_UTIL_GENERAL_H
#define TVMET_UTIL_GENERAL_H


/** forward */
namespace tvmet {
template<class T, int Rows, int Cols> class Matrix;
template<class T, int Sz> class Vector;
}

namespace tvmet {

namespace util {

/*
 * \defgroup _util_function
 * \brief Usefull utility functions
 */

/**
 * \fn Gemm(const Matrix<T, Rows, Cols>& m1, const Matrix<T, Rows, Cols>& m2, Matrix<T, Rows, Cols>& m3)
 * \brief General matrix matrix multiplication using loops.
 * \ingroup _util_function
 */
template<class T, int Rows, int Cols>
inline
void
Gemm(const Matrix<T, Rows, Cols>& m1, const Matrix<T, Rows, Cols>& m2,
     Matrix<T, Rows, Cols>& m3)
{
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      T sum(0);
      for (int k = 0; k < Cols; ++k) {
	sum += m1(i,k) * m2(k,j);
      }
      m3(i,j) = sum;
    }
  }
}


/**
 * \fn Gemv(const Matrix<T, Rows, Cols>& m, const Vector<T, Cols>& v, Vector<T, Cols>& v2)
 * \brief General matrix vector multiplication using loops.
 * \ingroup _util_function
 */
template<class T, int Rows, int Cols>
inline
void
Gemv(const Matrix<T, Rows, Cols>& m, const Vector<T, Cols>& v,
     Vector<T, Cols>& v2)
{
  for (int i = 0; i < Rows; ++i){
    v2(i) = T(0);	// clean up before use
    for (int j = 0; j < Cols; ++j) {
      v2(i) += m(i,j) * v(j);
      }
  }
}


/**
 * \fn Gevvmul(const Vector<T, Sz>& v1, const Vector<T, Sz>& v2, Vector<T, Sz>& v3)
 * \brief General vector vector elementwise multiplication using loop.
 * \ingroup _util_function
 */
template<class T, int Sz>
inline
void
Gevvmul(const Vector<T, Sz>& v1, const Vector<T, Sz>& v2,
       Vector<T, Sz>& v3)
{
  for(int i = 0; i < Sz; ++i)
    v3(i) = v1(i) * v2(i);
}


/**
 * \fn Gevvadd(const Vector<T, Sz>& v1, const Vector<T, Sz>& v2, Vector<T, Sz>& v3)
 * \brief General vector vector elementwise multiplication using loop.
 * \ingroup _util_function
 */
template<class T, int Sz>
inline
void
Gevvadd(const Vector<T, Sz>& v1, const Vector<T, Sz>& v2,
	Vector<T, Sz>& v3)
{
  for(int i = 0; i < Sz; ++i)
    v3(i) = v1(i) + v2(i);
}

} // namespace util

} // namespace tvmet

#endif // TVMET_UTIL_GENERAL_H

// Local Variables:
// mode:C++
// End:
