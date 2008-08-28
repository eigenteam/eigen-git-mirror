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

/** \defgroup flags
  * \ingroup Core_Module
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
  * Short version: means the expression might be vectorized
  *
  * Long version: means that the coefficients can be handled by packets
  * and start at a memory location whose alignment meets the requirements
  * of the present CPU architecture for optimized packet access. In the fixed-size
  * case, there is the additional condition that the total size of the coefficients
  * array is a multiple of the packet size, so that it is possible to access all the
  * coefficients by packets. In the dynamic-size case, there is no such condition
  * on the total size, so it might not be possible to access the few last coeffs
  * by packets.
  *
  * \note This bit can be set regardless of whether vectorization is actually enabled.
  *       To check for actual vectorizability, see \a ActualPacketAccessBit.
  */
const unsigned int PacketAccessBit = 0x8;

#ifdef EIGEN_VECTORIZE
/** \ingroup flags
  *
  * If vectorization is enabled (EIGEN_VECTORIZE is defined) this constant
  * is set to the value \a PacketAccessBit.
  *
  * If vectorization is not enabled (EIGEN_VECTORIZE is not defined) this constant
  * is set to the value 0.
  */
const unsigned int ActualPacketAccessBit = PacketAccessBit;
#else
const unsigned int ActualPacketAccessBit = 0x0;
#endif

/** \ingroup flags
  *
  * Short version: means the expression can be seen as 1D vector.
  *
  * Long version: means that one can access the coefficients
  * of this expression by coeff(int), and coeffRef(int) in the case of a lvalue expression. These
  * index-based access methods are guaranteed
  * to not have to do any runtime computation of a (row, col)-pair from the index, so that it
  * is guaranteed that whenever it is available, index-based access is at least as fast as
  * (row,col)-based access. Expressions for which that isn't possible don't have the LinearAccessBit.
  *
  * If both PacketAccessBit and LinearAccessBit are set, then the
  * packets of this expression can be accessed by packet(int), and writePacket(int) in the case of a
  * lvalue expression.
  *
  * Typically, all vector expressions have the LinearAccessBit, but there is one exception:
  * Product expressions don't have it, because it would be troublesome for vectorization, even when the
  * Product is a vector expression. Thus, vector Product expressions allow index-based coefficient access but
  * not index-based packet access, so they don't have the LinearAccessBit.
  */
const unsigned int LinearAccessBit = 0x10;

/** \ingroup flags
  *
  * Means that the underlying array of coefficients can be directly accessed. This means two things.
  * First, references to the coefficients must be available through coeffRef(int, int). This rules out read-only
  * expressions whose coefficients are computed on demand by coeff(int, int). Second, the memory layout of the
  * array of coefficients must be exactly the natural one suggested by rows(), cols(), stride(), and the RowMajorBit.
  * This rules out expressions such as DiagonalCoeffs, whose coefficients, though referencable, do not have
  * such a regular memory layout.
  */
const unsigned int DirectAccessBit = 0x20;

/** \ingroup flags
  *
  * means the first coefficient packet is guaranteed to be aligned */
const unsigned int AlignedBit = 0x40;

/** \ingroup flags
  *
  * means all diagonal coefficients are equal to 0 */
const unsigned int ZeroDiagBit = 0x80;

/** \ingroup flags
  *
  * means all diagonal coefficients are equal to 1 */
const unsigned int UnitDiagBit = 0x100;

/** \ingroup flags
  *
  * means the matrix is selfadjoint (M=M*). */
const unsigned int SelfAdjointBit = 0x200;

/** \ingroup flags
  *
  * means the strictly lower triangular part is 0 */
const unsigned int UpperTriangularBit = 0x400;

/** \ingroup flags
  *
  * means the strictly upper triangular part is 0 */
const unsigned int LowerTriangularBit = 0x800;

/** \ingroup flags
  *
  * means the expression includes sparse matrices and the sparse path has to be taken. */
const unsigned int SparseBit = 0x1000;

// list of flags that are inherited by default
const unsigned int HereditaryBits = RowMajorBit
                                  | EvalBeforeNestingBit
                                  | EvalBeforeAssigningBit
                                  | SparseBit;

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

enum { Aligned, Unaligned };
enum { ForceAligned, AsRequested };
enum { ConditionalJumpCost = 5 };
enum CornerType { TopLeft, TopRight, BottomLeft, BottomRight };
enum DirectionType { Vertical, Horizontal };
enum ProductEvaluationMode { NormalProduct, CacheFriendlyProduct, DiagonalProduct, SparseProduct };

enum {
  /** \internal Equivalent to a slice vectorization for fixed-size matrices having good alignment
    * and good size */
  InnerVectorization,
  /** \internal Vectorization path using a single loop plus scalar loops for the
    * unaligned boundaries */
  LinearVectorization,
  /** \internal Generic vectorization path using one vectorized loop per row/column with some
    * scalar loops to handle the unaligned boundaries */
  SliceVectorization,
  NoVectorization
};

enum {
  CompleteUnrolling,
  InnerUnrolling,
  NoUnrolling
};

enum {
  Dense   = 0,
  Sparse  = SparseBit
};

enum {
  ColMajor = 0,
  RowMajor = RowMajorBit
};

enum {
  NoDirectAccess = 0,
  HasDirectAccess = DirectAccessBit
};

const int FullyCoherentAccessPattern  = 0x1;
const int InnerCoherentAccessPattern  = 0x2 | FullyCoherentAccessPattern;
const int OuterCoherentAccessPattern  = 0x4 | InnerCoherentAccessPattern;
const int RandomAccessPattern         = 0x8 | OuterCoherentAccessPattern;

#endif // EIGEN_CONSTANTS_H
