// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_PERMUTATIONMATRIX_H
#define EIGEN_PERMUTATIONMATRIX_H

/** \nonstableyet
  * \class PermutationMatrix
  *
  * \brief Permutation matrix
  *
  * \param SizeAtCompileTime the number of rows/cols, or Dynamic
  * \param MaxSizeAtCompileTime the maximum number of rows/cols, or Dynamic. This optional parameter defaults to SizeAtCompileTime.
  *
  * This class represents a permutation matrix, internally stored as a vector of integers.
  * The convention followed here is the same as on <a href="http://en.wikipedia.org/wiki/Permutation_matrix">Wikipedia</a>,
  * namely: the matrix of permutation \a p is the matrix such that on each row \a i, the only nonzero coefficient is
  * in column p(i).
  *
  * \sa class DiagonalMatrix
  */
template<int SizeAtCompileTime, int MaxSizeAtCompileTime = SizeAtCompileTime> class PermutationMatrix;
template<typename PermutationType, typename MatrixType, int Side> struct ei_permut_matrix_product_retval;

template<int SizeAtCompileTime, int MaxSizeAtCompileTime>
struct ei_traits<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime> >
 : ei_traits<Matrix<int,SizeAtCompileTime,SizeAtCompileTime,0,MaxSizeAtCompileTime,MaxSizeAtCompileTime> >
{};

template<int SizeAtCompileTime, int MaxSizeAtCompileTime>
class PermutationMatrix : public AnyMatrixBase<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime> >
{
  public:

    typedef ei_traits<PermutationMatrix> Traits;
    typedef Matrix<int,SizeAtCompileTime,SizeAtCompileTime,0,MaxSizeAtCompileTime,MaxSizeAtCompileTime>
            DenseMatrixType;
    enum {
      Flags = Traits::Flags,
      CoeffReadCost = Traits::CoeffReadCost,
      RowsAtCompileTime = Traits::RowsAtCompileTime,
      ColsAtCompileTime = Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Traits::MaxColsAtCompileTime
    };
    typedef typename Traits::Scalar Scalar;

    typedef Matrix<int, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime, 1> IndicesType;

    inline PermutationMatrix()
    {
    }

    template<int OtherSize, int OtherMaxSize>
    inline PermutationMatrix(const PermutationMatrix<OtherSize, OtherMaxSize>& other)
      : m_indices(other.indices()) {}

    /** copy constructor. prevent a default copy constructor from hiding the other templated constructor */
    inline PermutationMatrix(const PermutationMatrix& other) : m_indices(other.indices()) {}

    /** generic constructor from expression of the indices */
    template<typename Other>
    explicit inline PermutationMatrix(const MatrixBase<Other>& other) : m_indices(other)
    {}

    template<int OtherSize, int OtherMaxSize>
    PermutationMatrix& operator=(const PermutationMatrix<OtherSize, OtherMaxSize>& other)
    {
      m_indices = other.indices();
      return *this;
    }

    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    PermutationMatrix& operator=(const PermutationMatrix& other)
    {
      m_indices = other.m_indices();
      return *this;
    }

    inline PermutationMatrix(int rows, int cols) : m_indices(rows)
    {
      ei_assert(rows == cols);
    }

    /** \returns the number of rows */
    inline int rows() const { return m_indices.size(); }

    /** \returns the number of columns */
    inline int cols() const { return m_indices.size(); }

    template<typename DenseDerived>
    void evalTo(MatrixBase<DenseDerived>& other) const
    {
      other.setZero();
      for (int i=0; i<rows();++i)
        other.coeffRef(i,m_indices.coeff(i)) = typename DenseDerived::Scalar(1);
    }

    DenseMatrixType toDenseMatrix() const
    {
      return *this;
    }

    const IndicesType& indices() const { return m_indices; }
    IndicesType& indices() { return m_indices; }

    /**** inversion and multiplication helpers to hopefully get RVO ****/

  protected:
    enum Inverse_t {Inverse};
    PermutationMatrix(Inverse_t, const PermutationMatrix& other)
      : m_indices(other.m_indices.size())
    {
      for (int i=0; i<rows();++i) m_indices.coeffRef(other.m_indices.coeff(i)) = i;
    }
    enum Product_t {Product};
    PermutationMatrix(Product_t, const PermutationMatrix& lhs, const PermutationMatrix& rhs)
      : m_indices(lhs.m_indices.size())
    {
      ei_assert(lhs.cols() == rhs.rows());
      for (int i=0; i<rows();++i) m_indices.coeffRef(i) = lhs.m_indices.coeff(rhs.m_indices.coeff(i));
    }

  public:
    inline PermutationMatrix inverse() const
    { return PermutationMatrix(Inverse, *this); }
    template<int OtherSize, int OtherMaxSize>
    inline PermutationMatrix operator*(const PermutationMatrix<OtherSize, OtherMaxSize>& other) const
    { return PermutationMatrix(Product, *this, other); }

  protected:

    IndicesType m_indices;
};

/** \returns the matrix with the permutation applied to the columns.
  */
template<typename Derived, int SizeAtCompileTime, int MaxSizeAtCompileTime>
inline const ei_permut_matrix_product_retval<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime>, Derived, OnTheRight>
operator*(const MatrixBase<Derived>& matrix,
          const PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime> &permutation)
{
  return ei_permut_matrix_product_retval
           <PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime>, Derived, OnTheRight>
           (permutation, matrix.derived());
}

/** \returns the matrix with the permutation applied to the rows.
  */
template<typename Derived, int SizeAtCompileTime, int MaxSizeAtCompileTime>
inline const ei_permut_matrix_product_retval
               <PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime>, Derived, OnTheLeft>
operator*(const PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime> &permutation,
          const MatrixBase<Derived>& matrix)
{
  return ei_permut_matrix_product_retval
           <PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime>, Derived, OnTheLeft>
           (permutation, matrix.derived());
}

template<typename PermutationType, typename MatrixType, int Side>
struct ei_traits<ei_permut_matrix_product_retval<PermutationType, MatrixType, Side> >
{
  typedef typename MatrixType::PlainMatrixType ReturnMatrixType;
};

template<typename PermutationType, typename MatrixType, int Side>
struct ei_permut_matrix_product_retval
 : public ReturnByValue<ei_permut_matrix_product_retval<PermutationType, MatrixType, Side> >
{
    typedef typename ei_cleantype<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;

    ei_permut_matrix_product_retval(const PermutationType& perm, const MatrixType& matrix)
      : m_permutation(perm), m_matrix(matrix)
    {}

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    template<typename Dest> inline void evalTo(Dest& dst) const
    {
      const int n = Side==OnTheLeft ? rows() : cols();
      for(int i = 0; i < n; ++i)
      {
        Block<
          Dest,
          Side==OnTheLeft ? 1 : Dest::RowsAtCompileTime,
          Side==OnTheRight ? 1 : Dest::ColsAtCompileTime
        >(dst, Side==OnTheRight ? m_permutation.indices().coeff(i) : i)

        =

        Block<
          MatrixTypeNestedCleaned,
          Side==OnTheLeft ? 1 : MatrixType::RowsAtCompileTime,
          Side==OnTheRight ? 1 : MatrixType::ColsAtCompileTime
        >(m_matrix, Side==OnTheLeft ? m_permutation.indices().coeff(i) : i);
      }
    }

  protected:
    const PermutationType& m_permutation;
    const typename MatrixType::Nested m_matrix;
};

#endif // EIGEN_PERMUTATIONMATRIX_H
