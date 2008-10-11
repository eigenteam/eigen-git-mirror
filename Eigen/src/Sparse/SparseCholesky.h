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

#ifndef EIGEN_SPARSECHOLESKY_H
#define EIGEN_SPARSECHOLESKY_H

enum SparseBackend {
  DefaultBackend,
  Taucs,
  Cholmod,
  SuperLU
};

enum {
  CompleteFactorization        = 0x0,  // full is the default
  IncompleteFactorization      = 0x1,
  MemoryEfficient              = 0x2,
  SupernodalMultifrontal       = 0x4,
  SupernodalLeftLooking        = 0x8


/*
  CholUseEigen    = 0x0,  // Eigen's impl is the default
  CholUseTaucs    = 0x2,
  CholUseCholmod  = 0x4*/
};

/** \ingroup Sparse_Module
  *
  * \class SparseCholesky
  *
  * \brief Standard Cholesky decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the Cholesky decomposition
  *
  * \sa class Cholesky, class CholeskyWithoutSquareRoot
  */
template<typename MatrixType, int Backend = DefaultBackend> class SparseCholesky
{
  protected:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef SparseMatrix<Scalar,Lower> CholMatrixType;

    enum {
      SupernodalFactorIsDirty      = 0x10000,
      MatrixLIsDirty               = 0x20000
    };

  public:

    SparseCholesky(const MatrixType& matrix, int flags = 0)
      : m_matrix(matrix.rows(), matrix.cols()), m_flags(flags), m_status(0)
    {
      compute(matrix);
    }

    inline const CholMatrixType& matrixL(void) const { return m_matrix; }

    /** \returns true if the matrix is positive definite */
    inline bool isPositiveDefinite(void) const { return m_isPositiveDefinite; }

    template<typename Derived>
    void solveInPlace(MatrixBase<Derived> &b) const;

    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store L
      * The strict upper part is not used and even not initialized.
      */
    CholMatrixType m_matrix;
    int m_flags;
    mutable int m_status;
    bool m_isPositiveDefinite;
};

/** Computes / recomputes the Cholesky decomposition A = LL^* = U^*U of \a matrix
  */
template<typename MatrixType, int Backend>
void SparseCholesky<MatrixType,Backend>::compute(const MatrixType& a)
{
  assert(a.rows()==a.cols());
  const int size = a.rows();
  m_matrix.resize(size, size);
  const RealScalar eps = ei_sqrt(precision<Scalar>());

  // allocate a temporary vector for accumulations
  AmbiVector<Scalar> tempVector(size);

  // TODO estimate the number of nnz
  m_matrix.startFill(a.nonZeros()*2);
  for (int j = 0; j < size; ++j)
  {
//     std::cout << j << "\n";
    Scalar x = ei_real(a.coeff(j,j));
    int endSize = size-j-1;

    // TODO estimate the number of non zero entries
//       float ratioLhs = float(lhs.nonZeros())/float(lhs.rows()*lhs.cols());
//       float avgNnzPerRhsColumn = float(rhs.nonZeros())/float(cols);
//       float ratioRes = std::min(ratioLhs * avgNnzPerRhsColumn, 1.f);

        // let's do a more accurate determination of the nnz ratio for the current column j of res
        //float ratioColRes = std::min(ratioLhs * rhs.innerNonZeros(j), 1.f);
        // FIXME find a nice way to get the number of nonzeros of a sub matrix (here an inner vector)
//         float ratioColRes = ratioRes;
//         if (ratioColRes>0.1)
//     tempVector.init(IsSparse);
    tempVector.init(IsDense);
    tempVector.setBounds(j+1,size);
    tempVector.setZero();
    // init with current matrix a
    {
      typename MatrixType::InnerIterator it(a,j);
      ++it; // skip diagonal element
      for (; it; ++it)
        tempVector.coeffRef(it.index()) = it.value();
    }
    for (int k=0; k<j+1; ++k)
    {
      typename MatrixType::InnerIterator it(m_matrix, k);
      while (it && it.index()<j)
        ++it;
      if (it && it.index()==j)
      {
        Scalar y = it.value();
        x -= ei_abs2(y);
        ++it; // skip j-th element, and process remaing column coefficients
        tempVector.restart();
        for (; it; ++it)
        {
          tempVector.coeffRef(it.index()) -= it.value() * y;
        }
      }
    }
    // copy the temporary vector to the respective m_matrix.col()
    // while scaling the result by 1/real(x)
    RealScalar rx = ei_sqrt(ei_real(x));
    m_matrix.fill(j,j) = rx;
    Scalar y = Scalar(1)/rx;
    for (typename AmbiVector<Scalar>::Iterator it(tempVector); it; ++it)
    {
      m_matrix.fill(it.index(), j) = it.value() * y;
    }
  }
  m_matrix.endFill();
}

template<typename MatrixType, int Backend>
template<typename Derived>
void SparseCholesky<MatrixType, Backend>::solveInPlace(MatrixBase<Derived> &b) const
{
  const int size = m_matrix.rows();
  ei_assert(size==b.rows());

  m_matrix.solveTriangularInPlace(b);
  m_matrix.adjoint().solveTriangularInPlace(b);
}

#endif // EIGEN_BASICSPARSECHOLESKY_H
