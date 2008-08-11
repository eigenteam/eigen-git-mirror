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
  int TriangularPart = ei_is_part<Lhs>::value ? -1  // this is to solve ambiguous specializations
                     : (int(Lhs::Flags) & LowerTriangularBit)
                     ? Lower
                     : (int(Lhs::Flags) & UpperTriangularBit)
                     ? Upper
                     : -1,
  int StorageOrder = int(Lhs::Flags) & RowMajorBit ? RowMajor : ColMajor
  >
struct ei_trisolve_selector;

// transform a Part xpr to a Flagged xpr
template<typename Lhs, unsigned int LhsMode, typename Rhs, int TriangularPart, int StorageOrder>
struct ei_trisolve_selector<Part<Lhs,LhsMode>,Rhs,TriangularPart,StorageOrder>
{
  static void run(const Part<Lhs,LhsMode>& lhs, Rhs& other)
  {
    ei_trisolve_selector<Flagged<Lhs,LhsMode,0>,Rhs>::run(lhs._expression(), other);
  }
};

// forward substitution, row-major
template<typename Lhs, typename Rhs>
struct ei_trisolve_selector<Lhs,Rhs,Lower,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, Rhs& other)
  {
    for(int c=0 ; c<other.cols() ; ++c)
    {
      if(!(Lhs::Flags & UnitDiagBit))
        other.coeffRef(0,c) = other.coeff(0,c)/lhs.coeff(0, 0);
      for(int i=1; i<lhs.rows(); ++i)
      {
        Scalar tmp = other.coeff(i,c) - ((lhs.row(i).start(i)) * other.col(c).start(i)).coeff(0,0);
        if (Lhs::Flags & UnitDiagBit)
          other.coeffRef(i,c) = tmp;
        else
          other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
      }
    }
  }
};

// backward substitution, row-major
template<typename Lhs, typename Rhs>
struct ei_trisolve_selector<Lhs,Rhs,Upper,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, Rhs& other)
  {
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      if(!(Lhs::Flags & UnitDiagBit))
        other.coeffRef(size-1,c) = other.coeff(size-1, c)/lhs.coeff(size-1, size-1);
      for(int i=size-2 ; i>=0 ; --i)
      {
        Scalar tmp = other.coeff(i,c)
                   - ((lhs.row(i).end(size-i-1)) * other.col(c).end(size-i-1)).coeff(0,0);
        if (Lhs::Flags & UnitDiagBit)
          other.coeffRef(i,c) = tmp;
        else
          other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
      }
    }
  }
};

// forward substitution, col-major
// FIXME the Lower and Upper specialization could be merged using a small helper class
// performing reflexions on the coordinates...
template<typename Lhs, typename Rhs>
struct ei_trisolve_selector<Lhs,Rhs,Lower,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  enum {PacketSize =  ei_packet_traits<Scalar>::size};

  static void run(const Lhs& lhs, Rhs& other)
  {
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      /* let's perform the inverse product per block of 4 columns such that we perfectly match
       * our optimized matrix * vector product.
       */
      int blockyEnd = (std::max(size-5,0)/4)*4;
      for(int i=0; i<blockyEnd;)
      {
        /* Let's process the 4x4 sub-matrix as usual.
         * btmp stores the diagonal coefficients used to update the remaining part of the result.
         */
        int startBlock = i;
        int endBlock = startBlock+4;
        Matrix<Scalar,4,1> btmp;
        for (;i<endBlock;++i)
        {
          if(!(Lhs::Flags & UnitDiagBit))
            other.coeffRef(i,c) /= lhs.coeff(i,i);
          int remainingSize = endBlock-i-1;
          if (remainingSize>0)
            other.col(c).block(i+1,remainingSize) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, i+1, i, remainingSize, 1);
          btmp.coeffRef(i-startBlock) = -other.coeffRef(i,c);
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
          size-endBlock, &(lhs.const_cast_derived().coeffRef(endBlock,startBlock)), lhs.stride(),
          btmp, &(other.coeffRef(endBlock,c)));
      }

      /* Now we have to process the remaining part as usual */
      int i;
      for(i=blockyEnd; i<size-1; ++i)
      {
        if(!(Lhs::Flags & UnitDiagBit))
          other.coeffRef(i,c) /= lhs.coeff(i,i);

        /* NOTE we cannot use lhs.col(i).end(size-i-1) because Part::coeffRef gets called by .col() to
         * get the address of the start of the row
         */
        other.col(c).end(size-i-1) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, i+1,i, size-i-1,1);
      }
      if(!(Lhs::Flags & UnitDiagBit))
        other.coeffRef(i,c) /= lhs.coeff(i,i);
    }
  }
};

// backward substitution, col-major
// see the previous specialization for details on the algorithm
template<typename Lhs, typename Rhs>
struct ei_trisolve_selector<Lhs,Rhs,Upper,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, Rhs& other)
  {
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      int blockyEnd = size-1 - (std::max(size-5,0)/4)*4;
      for(int i=size-1; i>blockyEnd;)
      {
        int startBlock = i;
        int endBlock = startBlock-4;
        Matrix<Scalar,4,1> btmp;
        /* Let's process the 4x4 sub-matrix as usual.
         * btmp stores the diagonal coefficients used to update the remaining part of the result.
         */
        for (; i>endBlock; --i)
        {
          if(!(Lhs::Flags & UnitDiagBit))
            other.coeffRef(i,c) /= lhs.coeff(i,i);
          int remainingSize = i-endBlock-1;
          if (remainingSize>0)
            other.col(c).block(endBlock+1,remainingSize) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, endBlock+1, i, remainingSize, 1);
          btmp.coeffRef(remainingSize) = -other.coeffRef(i,c);
        }

        ei_cache_friendly_product_colmajor_times_vector(
          endBlock+1, &(lhs.const_cast_derived().coeffRef(0,endBlock+1)), lhs.stride(),
          btmp, &(other.coeffRef(0,c)));
      }

      for(int i=blockyEnd; i>0; --i)
      {
        if(!(Lhs::Flags & UnitDiagBit))
          other.coeffRef(i,c) /= lhs.coeff(i,i);
        other.col(c).start(i) -= other.coeffRef(i,c) * Block<Lhs,Dynamic,1>(lhs, 0,i, i, 1);
      }
      if(!(Lhs::Flags & UnitDiagBit))
        other.coeffRef(0,c) /= lhs.coeff(0,0);
    }
  }
};

/** "in-place" version of MatrixBase::solveTriangular() where the result is written in \a other
  *
  * \sa solveTriangular()
  */
template<typename Derived>
template<typename OtherDerived>
void MatrixBase<Derived>::solveTriangularInPlace(MatrixBase<OtherDerived>& other) const
{
  ei_assert(derived().cols() == derived().rows());
  ei_assert(derived().cols() == other.rows());
  ei_assert(!(Flags & ZeroDiagBit));
  ei_assert(Flags & (UpperTriangularBit|LowerTriangularBit));

  ei_trisolve_selector<Derived, OtherDerived>::run(derived(), other.derived());
}

/** \returns the product of the inverse of \c *this with \a other, \a *this being triangular.
  *
  * This function computes the inverse-matrix matrix product inverse(\c *this) * \a other
  * It works as a forward (resp. backward) substitution if \c *this is an upper (resp. lower)
  * triangular matrix.
  *
  * It is required that \c *this be marked as either an upper or a lower triangular matrix, as
  * can be done by marked(), and as is automatically the case with expressions such as those returned
  * by extract().
  * Example: \include MatrixBase_marked.cpp
  * Output: \verbinclude MatrixBase_marked.out
  *
  * \sa marked(), extract()
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
