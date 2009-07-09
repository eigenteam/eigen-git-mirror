// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

template<typename Lhs, typename Rhs,
  int Mode, // Upper/Lower | UnitDiag
//   bool ConjugateLhs, bool ConjugateRhs,
  int UpLo = (Mode & LowerTriangularBit)
           ? LowerTriangular
           : (Mode & UpperTriangularBit)
           ? UpperTriangular
           : -1,
  int StorageOrder = int(Lhs::Flags) & RowMajorBit
  >
struct ei_triangular_solver_selector;

// forward substitution, row-major
template<typename Lhs, typename Rhs, int Mode, /*bool ConjugateLhs, bool ConjugateRhs,*/ int UpLo>
struct ei_triangular_solver_selector<Lhs,Rhs,Mode,/*ConjugateLhs,ConjugateRhs,*/UpLo,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, Rhs& other)
  {std::cerr << "here\n";
    #if NOTDEF
    const bool IsLowerTriangular = (UpLo==LowerTriangular);
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      const int PanelWidth = 4;
      for(int pi=IsLowerTriangular ? 0 : size;
          IsLowerTriangular ? pi<size : pi>0;
          IsLowerTriangular ? pi+=PanelWidth : pi-=PanelWidth)
      {
        int actualPanelWidth = std::min(IsLowerTriangular ? size - pi : pi, PanelWidth);
        int startBlock = IsLowerTriangular ? pi : pi-actualPanelWidth;
        int endBlock = IsLowerTriangular ? pi + actualPanelWidth : 0;

        if (pi > 0)
        {
          int r = IsLowerTriangular ? size - endBlock : startBlock; // remaining size
          ei_cache_friendly_product_colmajor_times_vector<false,false>(
            r,
            &(lhs.const_cast_derived().coeffRef(endBlock,startBlock)), lhs.stride(),
            other.col(c).segment(startBlock, actualPanelWidth),
            &(other.coeffRef(endBlock, c)),
            Scalar(-1));
        }

        for(int k=0; k<actualPanelWidth; ++k)
        {
          int i = IsLowerTriangular ? pi+k : pi-k-1;
          if(!(Mode & UnitDiagBit))
            other.coeffRef(i,c) /= lhs.coeff(i,i);

          int r = actualPanelWidth - k - 1; // remaining size
          if (r>0)
          {
            other.col(c).segment((IsLowerTriangular ? i+1 : i-r), r) -=
                  other.coeffRef(i,c)
                * Block<Lhs,Dynamic,1>(lhs, (IsLowerTriangular ? i+1 : i-r), i, r, 1);
          }
        }
      }
    }
    #else
    const bool IsLowerTriangular = (UpLo==LowerTriangular);
    const int size = lhs.cols();
    /* We perform the inverse product per block of 4 rows such that we perfectly match
     * our optimized matrix * vector product. blockyStart represents the number of rows
     * we have process first using the non-block version.
     */
    int blockyStart = (std::max(size-5,0)/4)*4;
    if (IsLowerTriangular)
      blockyStart = size - blockyStart;
    else
      blockyStart -= 1;
    for(int c=0 ; c<other.cols() ; ++c)
    {
      // process first rows using the non block version
      if(!(Mode & UnitDiagBit))
      {
        if (IsLowerTriangular)
          other.coeffRef(0,c) = other.coeff(0,c)/lhs.coeff(0, 0);
        else
          other.coeffRef(size-1,c) = other.coeff(size-1, c)/lhs.coeff(size-1, size-1);
      }
      for(int i=(IsLowerTriangular ? 1 : size-2); IsLowerTriangular ? i<blockyStart : i>blockyStart; i += (IsLowerTriangular ? 1 : -1) )
      {
        Scalar tmp = other.coeff(i,c)
          - (IsLowerTriangular ? ((lhs.row(i).start(i)) * other.col(c).start(i)).coeff(0,0)
                     : ((lhs.row(i).end(size-i-1)) * other.col(c).end(size-i-1)).coeff(0,0));
        if (Mode & UnitDiagBit)
          other.coeffRef(i,c) = tmp;
        else
          other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
      }

      // now let's process the remaining rows 4 at once
      for(int i=blockyStart; IsLowerTriangular ? i<size : i>0; )
      {
        int startBlock = i;
        int endBlock = startBlock + (IsLowerTriangular ? 4 : -4);

        /* Process the i cols times 4 rows block, and keep the result in a temporary vector */
        // FIXME use fixed size block but take care to small fixed size matrices...
        Matrix<Scalar,Dynamic,1> btmp(4);
        if (IsLowerTriangular)
          btmp = lhs.block(startBlock,0,4,i) * other.col(c).start(i);
        else
          btmp = lhs.block(i-3,i+1,4,size-1-i) * other.col(c).end(size-1-i);

        /* Let's process the 4x4 sub-matrix as usual.
         * btmp stores the diagonal coefficients used to update the remaining part of the result.
         */
        {
          Scalar tmp = other.coeff(startBlock,c)-btmp.coeff(IsLowerTriangular?0:3);
          if (Mode & UnitDiagBit)
            other.coeffRef(i,c) = tmp;
          else
            other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
        }

        i += IsLowerTriangular ? 1 : -1;
        for (;IsLowerTriangular ? i<endBlock : i>endBlock; i += IsLowerTriangular ? 1 : -1)
        {
          int remainingSize = IsLowerTriangular ? i-startBlock : startBlock-i;
          Scalar tmp = other.coeff(i,c)
            - btmp.coeff(IsLowerTriangular ? remainingSize : 3-remainingSize)
            - (   lhs.row(i).segment(IsLowerTriangular ? startBlock : i+1, remainingSize)
              * other.col(c).segment(IsLowerTriangular ? startBlock : i+1, remainingSize)).coeff(0,0);

          if (Mode & UnitDiagBit)
            other.coeffRef(i,c) = tmp;
          else
            other.coeffRef(i,c) = tmp/lhs.coeff(i,i);
        }
      }
    }
    #endif
  }
};

// Implements the following configurations:
//  - inv(LowerTriangular,         ColMajor) * Column vector
//  - inv(LowerTriangular,UnitDiag,ColMajor) * Column vector
//  - inv(UpperTriangular,         ColMajor) * Column vector
//  - inv(UpperTriangular,UnitDiag,ColMajor) * Column vector
template<typename Lhs, typename Rhs, int Mode, int UpLo>
struct ei_triangular_solver_selector<Lhs,Rhs,Mode,UpLo,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  enum { PacketSize =  ei_packet_traits<Scalar>::size };

  static void run(const Lhs& lhs, Rhs& other)
  {
    static const int PanelWidth = 4; // TODO make this a user definable constant
    static const bool IsLowerTriangular = (UpLo==LowerTriangular);
    const int size = lhs.cols();
    for(int c=0 ; c<other.cols() ; ++c)
    {
      for(int pi=IsLowerTriangular ? 0 : size;
          IsLowerTriangular ? pi<size : pi>0;
          IsLowerTriangular ? pi+=PanelWidth : pi-=PanelWidth)
      {
        int actualPanelWidth = std::min(IsLowerTriangular ? size - pi : pi, PanelWidth);
        int startBlock = IsLowerTriangular ? pi : pi-actualPanelWidth;
        int endBlock = IsLowerTriangular ? pi + actualPanelWidth : 0;

        for(int k=0; k<actualPanelWidth; ++k)
        {
          int i = IsLowerTriangular ? pi+k : pi-k-1;
          if(!(Mode & UnitDiagBit))
            other.coeffRef(i,c) /= lhs.coeff(i,i);

          int r = actualPanelWidth - k - 1; // remaining size
          if (r>0)
          {
            other.col(c).segment((IsLowerTriangular ? i+1 : i-r), r) -=
                  other.coeffRef(i,c)
                * Block<Lhs,Dynamic,1>(lhs, (IsLowerTriangular ? i+1 : i-r), i, r, 1);
          }
        }
        int r = IsLowerTriangular ? size - endBlock : startBlock; // remaining size
        if (r > 0)
        {
          ei_cache_friendly_product_colmajor_times_vector<false,false>(
            r,
            &(lhs.const_cast_derived().coeffRef(endBlock,startBlock)), lhs.stride(),
            other.col(c).segment(startBlock, actualPanelWidth),
            &(other.coeffRef(endBlock, c)),
            Scalar(-1));
        }
      }
    }
  }
};

/** "in-place" version of MatrixBase::solveTriangular() where the result is written in \a other
  *
  * \nonstableyet
  *
  * \warning The parameter is only marked 'const' to make the C++ compiler accept a temporary expression here.
  * This function will const_cast it, so constness isn't honored here.
  *
  * See MatrixBase:solveTriangular() for the details.
  */
template<typename MatrixType, unsigned int Mode>
template<typename RhsDerived>
void TriangularView<MatrixType,Mode>::solveInPlace(const MatrixBase<RhsDerived>& _rhs) const
{
  RhsDerived& rhs = _rhs.const_cast_derived();
  ei_assert(cols() == rows());
  ei_assert(cols() == rhs.rows());
  ei_assert(!(Mode & ZeroDiagBit));
  ei_assert(Mode & (UpperTriangularBit|LowerTriangularBit));

  enum { copy = ei_traits<RhsDerived>::Flags & RowMajorBit };

  typedef typename ei_meta_if<copy,
    typename ei_plain_matrix_type_column_major<RhsDerived>::type, RhsDerived&>::ret RhsCopy;
  RhsCopy rhsCopy(rhs);

  ei_triangular_solver_selector<MatrixType, typename ei_unref<RhsCopy>::type, Mode>::run(_expression(), rhsCopy);

  if (copy)
    rhs = rhsCopy;
}

/** \returns the product of the inverse of \c *this with \a other, \a *this being triangular.
  *
  * \nonstableyet
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
  * \sa solveTriangularInPlace()
  */
template<typename Derived, unsigned int Mode>
template<typename RhsDerived>
typename ei_plain_matrix_type_column_major<RhsDerived>::type
TriangularView<Derived,Mode>::solve(const MatrixBase<RhsDerived>& rhs) const
{
  typename ei_plain_matrix_type_column_major<RhsDerived>::type res(rhs);
  solveInPlace(res);
  return res;
}

#endif // EIGEN_SOLVETRIANGULAR_H
