// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SOLVETRIANGULAR_H
#define EIGEN_SOLVETRIANGULAR_H

template<typename XprType> struct ei_is_part { enum {value=false}; };
template<typename XprType, unsigned int Mode> struct ei_is_part<Part<XprType,Mode> > { enum {value=true}; };

template<typename Lhs, typename Rhs,
  int TriangularPart = (int(Lhs::Flags) & LowerTriangularBit)
                     ? Lower
                     : (int(Lhs::Flags) & UpperTriangularBit)
                     ? Upper
                     : -1,
  int StorageOrder = ei_is_part<Lhs>::value ? -1  // this is to solve ambiguous specializations
                   : int(Lhs::Flags) & (RowMajorBit|SparseBit)
  >
struct ei_solve_triangular_selector;

// transform a Part xpr to a Flagged xpr
template<typename Lhs, unsigned int LhsMode, typename Rhs, int UpLo, int StorageOrder>
struct ei_solve_triangular_selector<Part<Lhs,LhsMode>,Rhs,UpLo,StorageOrder>
{
  static void run(const Part<Lhs,LhsMode>& lhs, Rhs& other)
  {
    ei_solve_triangular_selector<Flagged<Lhs,LhsMode,0>,Rhs>::run(lhs._expression(), other);
  }
};

// forward substitution, row-major
template<typename Lhs, typename Rhs, int UpLo>
struct ei_solve_triangular_selector<Lhs,Rhs,UpLo,RowMajor|IsDense>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, Rhs& other)
  {
    const bool IsLower = (UpLo==Lower);
    const int size = lhs.cols();
    /* We perform the inverse product per block of 4 rows such that we perfectly match
     * our optimized matrix * vector product. blockyStart represents the number of rows
     * we have process first using the non-block version.
     */
    int blockyStart = (std::max(size-5,0)/4)*4;
    if (IsLower)
      blockyStart = size - blockyStart;
    else
      blockyStart -= 1;
    for(int c=0 ; c<other.cols() ; ++c)
    {
      // process first rows using the non block version
      if(!(Lhs::Flags & UnitDiagBit))
      {
        if (IsLower)
          other.coeffRef(0,c) = other.coeff(0,c)/lhs.coeff(0, 0);
        else
          other.coeffRef(size-1,c) = other.coeff(size-1, c)/lhs.coeff(size-1, size-1);
      }
      for(int i=(IsLower ? 1 : size-2); IsLower ? i<blockyStart : i>blockyStart; i += (IsLower ? 1 : -1) )
      {
        Scalar tmp = other.coeff(i,c)
          - (IsLower ? ((lhs.row(i).start(i)) * other.col(c).start(i)).coeff(0,0)
                     : ((lhs.row(i).end(size-i-1)) * other.col(c).end(size-i-1)).coeff(0,0));
        if (Lhs::Flags & UnitDiagBit)
          other.coeffRef(i,c) = tmp;
        else
          other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
      }

      // now let process the remaining rows 4 at once
      for(int i=blockyStart; IsLower ? i<size : i>0; )
      {
        int startBlock = i;
        int endBlock = startBlock + (IsLower ? 4 : -4);
        
        /* Process the i cols times 4 rows block, and keep the result in a temporary vector */
        // FIXME use fixed size block but take care to small fixed size matrices...
        Matrix<Scalar,Dynamic,1> btmp(4);
        if (IsLower)
          btmp = lhs.block(startBlock,0,4,i) * other.col(c).start(i);
        else
          btmp = lhs.block(i-3,i+1,4,size-1-i) * other.col(c).end(size-1-i);
        
        /* Let's process the 4x4 sub-matrix as usual.
         * btmp stores the diagonal coefficients used to update the remaining part of the result.
         */
        {
          Scalar tmp = other.coeff(startBlock,c)-btmp.coeff(IsLower?0:3);
          if (Lhs::Flags & UnitDiagBit)
            other.coeffRef(i,c) = tmp;
          else
            other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
        }

        i += IsLower ? 1 : -1;
        for (;IsLower ? i<endBlock : i>endBlock; i += IsLower ? 1 : -1)
        {
          int remainingSize = IsLower ? i-startBlock : startBlock-i;
          Scalar tmp = other.coeff(i,c)
            - btmp.coeff(IsLower ? remainingSize : 3-remainingSize)
            - (   lhs.row(i).block(IsLower ? startBlock : i+1, remainingSize)
              * other.col(c).block(IsLower ? startBlock : i+1, remainingSize)).coeff(0,0);

          if (Lhs::Flags & UnitDiagBit)
            other.coeffRef(i,c) = tmp;
          else
            other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
        }
      }
    }
  }
};

// Implements the following configurations:
//  - inv(Lower,         ColMajor) * Column vector
//  - inv(Lower,UnitDiag,ColMajor) * Column vector
//  - inv(Upper,         ColMajor) * Column vector
//  - inv(Upper,UnitDiag,ColMajor) * Column vector
template<typename Lhs, typename Rhs, int UpLo>
struct ei_solve_triangular_selector<Lhs,Rhs,UpLo,ColMajor|IsDense>
{
  typedef typename Rhs::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  enum { PacketSize =  ei_packet_traits<Scalar>::size };

  static void run(const Lhs& lhs, Rhs& other)
  {
    static const bool IsLower = (UpLo==Lower);
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      /* let's perform the inverse product per block of 4 columns such that we perfectly match
       * our optimized matrix * vector product. blockyEnd represents the number of rows
       * we can process using the block version.
       */
      int blockyEnd = (std::max(size-5,0)/4)*4;
      if (!IsLower)
        blockyEnd = size-1 - blockyEnd;
      for(int i=IsLower ? 0 : size-1; IsLower ? i<blockyEnd : i>blockyEnd;)
      {
        /* Let's process the 4x4 sub-matrix as usual.
         * btmp stores the diagonal coefficients used to update the remaining part of the result.
         */
        int startBlock = i;
        int endBlock = startBlock + (IsLower ? 4 : -4);
        Matrix<Scalar,4,1> btmp;
        for (;IsLower ? i<endBlock : i>endBlock;
             i += IsLower ? 1 : -1)
        {
          if(!(Lhs::Flags & UnitDiagBit))
            other.coeffRef(i,c) /= lhs.coeff(i,i);
          int remainingSize = IsLower ? endBlock-i-1 : i-endBlock-1;
          if (remainingSize>0)
            other.col(c).block((IsLower ? i : endBlock) + 1, remainingSize) -=
                other.coeffRef(i,c)
              * Block<Lhs,Dynamic,1>(lhs, (IsLower ? i : endBlock) + 1, i, remainingSize, 1);
          btmp.coeffRef(IsLower ? i-startBlock : remainingSize) = -other.coeffRef(i,c);
        }

        /* Now we can efficiently update the remaining part of the result as a matrix * vector product.
         * NOTE in order to reduce both compilation time and binary size, let's directly call
         * the fast product implementation. It is equivalent to the following code:
         *   other.col(c).end(size-endBlock) += (lhs.block(endBlock, startBlock, size-endBlock, endBlock-startBlock)
         *                                       * other.col(c).block(startBlock,endBlock-startBlock)).lazy();
         */
        // FIXME this is cool but what about conjugate/adjoint expressions ? do we want to evaluate them ?
        // this is a more general problem though.
        ei_cache_friendly_product_colmajor_times_vector(
          IsLower ? size-endBlock : endBlock+1,
          &(lhs.const_cast_derived().coeffRef(IsLower ? endBlock : 0, IsLower ? startBlock : endBlock+1)),
          lhs.stride(),
          btmp, &(other.coeffRef(IsLower ? endBlock : 0, c)));
      }

      /* Now we have to process the remaining part as usual */
      int i;
      for(i=blockyEnd; IsLower ? i<size-1 : i>0; i += (IsLower ? 1 : -1) )
      {
        if(!(Lhs::Flags & UnitDiagBit))
          other.coeffRef(i,c) /= lhs.coeff(i,i);

        /* NOTE we cannot use lhs.col(i).end(size-i-1) because Part::coeffRef gets called by .col() to
         * get the address of the start of the row
         */
        if(IsLower)
          other.col(c).end(size-i-1) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, i+1,i, size-i-1,1);
        else
          other.col(c).start(i) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, 0,i, i, 1);
      }
      if(!(Lhs::Flags & UnitDiagBit))
        other.coeffRef(i,c) /= lhs.coeff(i,i);
    }
  }
};

/** "in-place" version of MatrixBase::solveTriangular() where the result is written in \a other
  *
  * See MatrixBase:solveTriangular() for the details.
  */
template<typename Derived>
template<typename OtherDerived>
void MatrixBase<Derived>::solveTriangularInPlace(MatrixBase<OtherDerived>& other) const
{
  ei_assert(derived().cols() == derived().rows());
  ei_assert(derived().cols() == other.rows());
  ei_assert(!(Flags & ZeroDiagBit));
  ei_assert(Flags & (UpperTriangularBit|LowerTriangularBit));

  ei_solve_triangular_selector<Derived, OtherDerived>::run(derived(), other.derived());
}

/** \returns the product of the inverse of \c *this with \a other, \a *this being triangular.
  *
  * This function computes the inverse-matrix matrix product inverse(\c *this) * \a other.
  * The matrix \c *this must be triangular and invertible (i.e., all the coefficients of the
  * diagonal must be non zero). It works as a forward (resp. backward) substitution if \c *this
  * is an upper (resp. lower) triangular matrix.
  *
  * It is required that \c *this be marked as either an upper or a lower triangular matrix, which
  * can be done by marked(), and that is automatically the case with expressions such as those returned
  * by extract().
  * 
  * \addexample SolveTriangular \label How to solve a triangular system (aka. how to multiply the inverse of a triangular matrix by another one)
  * 
  * Example: \include MatrixBase_marked.cpp
  * Output: \verbinclude MatrixBase_marked.out
  * 
  * This function is essentially a wrapper to the faster solveTriangularInPlace() function creating
  * a temporary copy of \a other, calling solveTriangularInPlace() on the copy and returning it.
  * Therefore, if \a other is not needed anymore, it is quite faster to call solveTriangularInPlace()
  * instead of solveTriangular().
  * 
  * For users coming from BLAS, this function (and more specifically solveTriangularInPlace()) offer
  * all the operations supported by the \c *TRSV and \c *TRSM BLAS routines.
  *
  * \b Tips: to perform a \em "right-inverse-multiply" you can simply transpose the operation, e.g.:
  * \code
  * M * T^1  <=>  T.transpose().solveTriangularInPlace(M.transpose());
  * \endcode
  * 
  * \sa solveTriangularInPlace(), marked(), extract()
  */
template<typename Derived>
template<typename OtherDerived>
typename OtherDerived::Eval MatrixBase<Derived>::solveTriangular(const MatrixBase<OtherDerived>& other) const
{
  typename OtherDerived::Eval res(other);
  solveTriangularInPlace(res);
  return res;
}

#endif // EIGEN_SOLVETRIANGULAR_H
