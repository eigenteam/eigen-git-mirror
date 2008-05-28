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

// matrix/expression flags
const unsigned int RowMajorBit = 0x1;
const unsigned int EvalBeforeNestingBit = 0x2;  ///< means the expression should be evaluated by the calling expression
const unsigned int EvalBeforeAssigningBit = 0x4;///< means the expression should be evaluated before any assignement
const unsigned int LargeBit = 0x8;
#ifdef EIGEN_VECTORIZE
const unsigned int VectorizableBit = 0x10;  ///< means the expression might be vectorized
#else
const unsigned int VectorizableBit = 0x0;
#endif
const unsigned int Like1DArrayBit = 0x20;   ///< means the expression can be seen as 1D vector (used for explicit vectorization)
const unsigned int ZeroDiagBit = 0x40;      ///< means all diagonal coefficients are equal to 0
const unsigned int UnitDiagBit = 0x80;      ///< means all diagonal coefficients are equal to 1
const unsigned int SelfAdjointBit = 0x100;  ///< means the matrix is selfadjoint (M=M*).
const unsigned int UpperTriangularBit = 0x200;    ///< means the strictly triangular lower part is 0
const unsigned int LowerTriangularBit = 0x400;    ///< means the strictly triangular upper part is 0
const unsigned int DirectAccessBit = 0x800; ///< means the underlying matrix data can be direclty accessed
const unsigned int NestByValueBit = 0x1000;   ///< means the expression should be copied by value when nested

// list of flags that are inherited by default
const unsigned int HereditaryBits = RowMajorBit
                                  | EvalBeforeNestingBit
                                  | EvalBeforeAssigningBit
                                  | LargeBit;

// Possible values for the Mode parameter of part() and of extract()
const unsigned int Upper = UpperTriangularBit;
const unsigned int StrictlyUpper = UpperTriangularBit | ZeroDiagBit;
const unsigned int Lower = LowerTriangularBit;
const unsigned int StrictlyLower = LowerTriangularBit | ZeroDiagBit;

// additional possible values for the Mode parameter of part()
const unsigned int SelfAdjoint = SelfAdjointBit;

// additional possible values for the Mode parameter of extract()
const unsigned int UnitUpper = UpperTriangularBit | UnitDiagBit;
const unsigned int UnitLower = LowerTriangularBit | UnitDiagBit;



enum { Aligned=0, UnAligned=1 };
enum { ConditionalJumpCost = 5 };
enum CornerType { TopLeft, TopRight, BottomLeft, BottomRight };
enum DirectionType { Vertical, Horizontal };
enum ProductEvaluationMode { NormalProduct, CacheFriendlyProduct, LazyProduct};

#endif // EIGEN_CONSTANTS_H
