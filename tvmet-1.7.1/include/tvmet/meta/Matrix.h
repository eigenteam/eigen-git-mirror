/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: Matrix.h,v 1.15 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_META_MATRIX_H
#define TVMET_META_MATRIX_H

#include <tvmet/NumericTraits.h>
#include <tvmet/xpr/Null.h>

namespace tvmet {

namespace meta {


/**
 * \class Matrix Matrix.h "tvmet/meta/Matrix.h"
 * \brief Meta %Matrix class using expression and meta templates.
 */
template<int Rows, int Cols,
	 int M=0, int N=0>
class Matrix
{
  Matrix();
  Matrix(const Matrix&);
  Matrix& operator=(const Matrix&);

private:
  enum {
    doRows = (M < Rows - 1) ? 1 : 0,	/**< recursive counter Rows. */
    doCols = (N < Cols - 1) ? 1 : 0	/**< recursive counter Cols. */
  };

public:
  /** assign an expression on columns on given row using the functional assign_fn. */
  template<class Dest, class Src, class Assign>
  static inline
  void assign2(Dest& lhs, const Src& rhs, const Assign& assign_fn) {
    assign_fn.apply_on(lhs(M, N), rhs(M, N));
    Matrix<Rows * doCols, Cols * doCols,
           M * doCols, (N+1) * doCols>::assign2(lhs, rhs, assign_fn);
  }

  /** assign an expression on row-wise using the functional assign_fn. */
  template<class Dest, class Src, class Assign>
  static inline
  void assign(Dest& lhs, const Src& rhs, const Assign& assign_fn) {
    Matrix<Rows, Cols,
           M, 0>::assign2(lhs, rhs, assign_fn);
    Matrix<Rows * doRows, Cols * doRows,
          (M+1) * doRows, 0>::assign(lhs, rhs, assign_fn);
  }

  /** evaluate a given matrix expression, column wise. */
  template<class E>
  static inline
  bool all_elements2(const E& e) {
    if(!e(M, N)) return false;
    return Matrix<Rows * doCols, Cols * doCols,
                  M * doCols, (N+1) * doCols>::all_elements2(e);
  }

  /** evaluate a given matrix expression, row wise. */
  template<class E>
  static inline
  bool all_elements(const E& e) {
    if(!Matrix<Rows, Cols, M, 0>::all_elements2(e) ) return false;
    return Matrix<Rows * doRows, Cols * doRows,
                 (M+1) * doRows, 0>::all_elements(e);
  }

  /** evaluate a given matrix expression, column wise. */
  template<class E>
  static inline
  bool any_elements2(const E& e) {
    if(e(M, N)) return true;
    return Matrix<Rows * doCols, Cols * doCols,
                  M * doCols, (N+1) * doCols>::any_elements2(e);
  }

  /** evaluate a given matrix expression, row wise. */
  template<class E>
  static inline
  bool any_elements(const E& e) {
    if(Matrix<Rows, Cols, M, 0>::any_elements2(e) ) return true;
    return Matrix<Rows * doRows, Cols * doRows,
                 (M+1) * doRows, 0>::any_elements(e);
  }

  /** trace a given matrix expression. */
  template<class E>
  static inline
  typename E::value_type
  trace(const E& e) {
    return e(M, N)
      + Matrix<Rows * doCols, Cols * doCols,
              (M+1) * doCols, (N+1) * doCols>::trace(e);
  }

};


/**
 * \class Matrix<0, 0, 0, 0> Matrix.h "tvmet/meta/Matrix.h"
 * \brief Meta %Matrix specialized for recursion.
 */
template<>
class Matrix<0, 0, 0, 0>
{
  Matrix();
  Matrix(const Matrix&);
  Matrix& operator=(const Matrix&);

public:
  template<class Dest, class Src, class Assign>
  static inline void assign2(Dest&, const Src&, const Assign&) { }

  template<class Dest, class Src, class Assign>
  static inline void assign(Dest&, const Src&, const Assign&) { }

  template<class E>
  static inline bool all_elements2(const E&) { return true; }

  template<class E>
  static inline bool all_elements(const E&) { return true; }

  template<class E>
  static inline bool any_elements2(const E&) { return false; }

  template<class E>
  static inline bool any_elements(const E&) { return false; }

  template<class E>
  static inline XprNull trace(const E&) { return XprNull(); }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_MATRIX_H */

// Local Variables:
// mode:C++
// End:
