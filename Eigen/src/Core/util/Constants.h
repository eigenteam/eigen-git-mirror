// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_CONSTANTS_H
#define EIGEN_CONSTANTS_H

const int Dynamic = 10000;

/** \defgroup flags */

/** \name flags
  *
  * These are the possible bits which can be OR'ed to constitute the flags of a matrix or
  * expression.
  *
  * \sa MatrixBase::Flags
  */

/** \ingroup flags
  *
  * for a matrix, this means that the storage order is row-major.
  * If this bit is not set, the storage order is column-major.
  * For an expression, this determines the storage order of
  * the matrix created by evaluation of that expression. */
const unsigned int RowMajorBit = 0x1;

/** \ingroup flags
  *
  * means the expression should be evaluated by the calling expression */
const unsigned int EvalBeforeNestingBit = 0x2;

/** \ingroup flags
  *
  * means the expression should be evaluated before any assignement */
const unsigned int EvalBeforeAssigningBit = 0x4;

/** \ingroup flags
  *
  * currently unused. Means the matrix probably has a very big size.
  * Could eventually be used as a hint to determine which algorithms
  * to use. */
const unsigned int LargeBit = 0x8;

#ifdef EIGEN_VECTORIZE
/** \ingroup flags
  *
  * means the expression might be vectorized */
const unsigned int VectorizableBit = 0x10;
#else
const unsigned int VectorizableBit = 0x0;
#endif

/** \ingroup flags
  *
  * means the expression can be seen as 1D vector (used for explicit vectorization) */
const unsigned int Like1DArrayBit = 0x20;

/** \ingroup flags
  *
  * means all diagonal coefficients are equal to 0 */
const unsigned int ZeroDiagBit = 0x40;

/** \ingroup flags
  *
  * means all diagonal coefficients are equal to 1 */
const unsigned int UnitDiagBit = 0x80;

/** \ingroup flags
  *
  * means the matrix is selfadjoint (M=M*). */
const unsigned int SelfAdjointBit = 0x100;

/** \ingroup flags
  *
  * means the strictly lower triangular part is 0 */
const unsigned int UpperTriangularBit = 0x200;

/** \ingroup flags
  *
  * means the strictly upper triangular part is 0 */
const unsigned int LowerTriangularBit = 0x400;

/** \ingroup flags
  *
  * means the underlying matrix data can be direclty accessed (contrary to certain
  * expressions where the matrix coefficients need to be computed rather than just read from
  * memory) */
const unsigned int DirectAccessBit = 0x800;

/** \ingroup flags
  *
  * means the object is just an array of scalars, and operations on it are regarded as operations
  * on every of these scalars taken separately.
  */
const unsigned int ArrayBit = 0x1000;

// list of flags that are inherited by default
const unsigned int HereditaryBits = RowMajorBit
                                  | EvalBeforeNestingBit
                                  | EvalBeforeAssigningBit
                                  | LargeBit
                                  | ArrayBit;

// Possible values for the Mode parameter of part() and of extract()
const unsigned int Upper = UpperTriangularBit;
const unsigned int StrictlyUpper = UpperTriangularBit | ZeroDiagBit;
const unsigned int Lower = LowerTriangularBit;
const unsigned int StrictlyLower = LowerTriangularBit | ZeroDiagBit;
const unsigned int SelfAdjoint = SelfAdjointBit;

// additional possible values for the Mode parameter of extract()
const unsigned int UnitUpper = UpperTriangularBit | UnitDiagBit;
const unsigned int UnitLower = LowerTriangularBit | UnitDiagBit;
const unsigned int Diagonal = Upper | Lower;


enum { Aligned=0, UnAligned=1 };
enum { ConditionalJumpCost = 5 };
enum CornerType { TopLeft, TopRight, BottomLeft, BottomRight };
enum DirectionType { Vertical, Horizontal };
enum ProductEvaluationMode { NormalProduct, CacheFriendlyProduct, DiagonalProduct, LazyProduct};


#endif // EIGEN_CONSTANTS_H
