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

#ifndef EIGEN_BASICSPARSECHOLESKY_H
#define EIGEN_BASICSPARSECHOLESKY_H

/** \ingroup Sparse_Module
  *
  * \class BasicSparseCholesky
  *
  * \brief Standard Cholesky decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the Cholesky decomposition
  *
  * \sa class Cholesky, class CholeskyWithoutSquareRoot
  */
template<typename MatrixType> class BasicSparseCholesky
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1
    };

  public:

    BasicSparseCholesky(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols())
    {
      compute(matrix);
    }

    inline const MatrixType& matrixL(void) const { return m_matrix; }

    /** \returns true if the matrix is positive definite */
    inline bool isPositiveDefinite(void) const { return m_isPositiveDefinite; }

//     template<typename Derived>
//     typename Derived::Eval solve(const MatrixBase<Derived> &b) const;

    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store L
      * The strict upper part is not used and even not initialized.
      */
    MatrixType m_matrix;
    bool m_isPositiveDefinite;

    struct ListEl
    {
      int next;
      int index;
      Scalar value;
    };
};

/** Computes / recomputes the Cholesky decomposition A = LL^* = U^*U of \a matrix
  */
#ifdef IGEN_BASICSPARSECHOLESKY_H
template<typename MatrixType>
void BasicSparseCholesky<MatrixType>::compute(const MatrixType& a)
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


#else

template<typename MatrixType>
void BasicSparseCholesky<MatrixType>::compute(const MatrixType& a)
{
  assert(a.rows()==a.cols());
  const int size = a.rows();
  m_matrix.resize(size, size);
  const RealScalar eps = ei_sqrt(precision<Scalar>());

  // allocate a temporary buffer
  Scalar* buffer = new Scalar[size*2];


  m_matrix.startFill(a.nonZeros()*2);

//   RealScalar x;
//   x = ei_real(a.coeff(0,0));
//   m_isPositiveDefinite = x > eps && ei_isMuchSmallerThan(ei_imag(a.coeff(0,0)), RealScalar(1));
//   m_matrix.fill(0,0) = ei_sqrt(x);
//   m_matrix.col(0).end(size-1) = a.row(0).end(size-1).adjoint() / ei_real(m_matrix.coeff(0,0));
  for (int j = 0; j < size; ++j)
  {
//     std::cout << j << " " << std::flush;
//     Scalar tmp = ei_real(a.coeff(j,j));
//     if (j>0)
//       tmp -= m_matrix.row(j).start(j).norm2();
//     x = ei_real(tmp);
//     std::cout << "x = " << x << "\n";
//     if (x < eps || (!ei_isMuchSmallerThan(ei_imag(tmp), RealScalar(1))))
//     {
//       m_isPositiveDefinite = false;
//       return;
//     }
//     m_matrix.fill(j,j) = x = ei_sqrt(x);

    Scalar x = ei_real(a.coeff(j,j));
//     if (j>0)
//       x -= m_matrix.row(j).start(j).norm2();
//     RealScalar rx = ei_sqrt(ei_real(x));
//     m_matrix.fill(j,j) = rx;
    int endSize = size-j-1;
    /*if (endSize>0)*/ {
      // Note that when all matrix columns have good alignment, then the following
      // product is guaranteed to be optimal with respect to alignment.
//       m_matrix.col(j).end(endSize) =
//         (m_matrix.block(j+1, 0, endSize, j) * m_matrix.row(j).start(j).adjoint()).lazy();

      // FIXME could use a.col instead of a.row
//       m_matrix.col(j).end(endSize) = (a.row(j).end(endSize).adjoint()
//         - m_matrix.col(j).end(endSize) ) / x;

      // make sure to call innerSize/outerSize since we fake the storage order.




      // estimate the number of non zero entries
//       float ratioLhs = float(lhs.nonZeros())/float(lhs.rows()*lhs.cols());
//       float avgNnzPerRhsColumn = float(rhs.nonZeros())/float(cols);
//       float ratioRes = std::min(ratioLhs * avgNnzPerRhsColumn, 1.f);


//       for (int j1=0; j1<cols; ++j1)
      {
        // let's do a more accurate determination of the nnz ratio for the current column j of res
        //float ratioColRes = std::min(ratioLhs * rhs.innerNonZeros(j), 1.f);
        // FIXME find a nice way to get the number of nonzeros of a sub matrix (here an inner vector)
//         float ratioColRes = ratioRes;
//         if (ratioColRes>0.1)
        if (true)
        {
          // dense path, the scalar * columns products are accumulated into a dense column
          Scalar* __restrict__ tmp = buffer;
          // set to zero
          for (int k=j+1; k<size; ++k)
            tmp[k] = 0;
          // init with current matrix a
          {
            typename MatrixType::InnerIterator it(a,j);
            ++it;
            for (; it; ++it)
              tmp[it.index()] = it.value();
          }
          for (int k=0; k<j+1; ++k)
          {
//             Scalar y = m_matrix.coeff(j,k);
//             if (!ei_isMuchSmallerThan(ei_abs(y),Scalar(1)))
//             {
            typename MatrixType::InnerIterator it(m_matrix, k);
            while (it && it.index()<j)
              ++it;
            if (it && it.index()==j)
            {
              Scalar y = it.value();
              x -= ei_abs2(y);
//               if (!ei_isMuchSmallerThan(ei_abs(y),Scalar(0.1)))
              {
                ++it;
                for (; it; ++it)
                {
                  tmp[it.index()] -= it.value() * y;
                }
              }
            }
          }
          // copy the temporary to the respective m_matrix.col()
          RealScalar rx = ei_sqrt(ei_real(x));
          m_matrix.fill(j,j) = rx;
          Scalar y = Scalar(1)/rx;
          for (int k=j+1; k<size; ++k)
            //if (tmp[k]!=0)
            if (!ei_isMuchSmallerThan(ei_abs(tmp[k]),Scalar(0.01)))
              m_matrix.fill(k, j) = tmp[k]*y;
        }
        else
        {
          ListEl* __restrict__ tmp = reinterpret_cast<ListEl*>(buffer);
          // sparse path, the scalar * columns products are accumulated into a linked list
          int tmp_size = 0;
          int tmp_start = -1;

          {
            int tmp_el = tmp_start;
            typename MatrixType::InnerIterator it(a,j);
            if (it)
            {
              ++it;
              for (; it; ++it)
              {
                Scalar v = it.value();
                int id = it.index();
                if (tmp_size==0)
                {
                  tmp_start = 0;
                  tmp_el = 0;
                  tmp_size++;
                  tmp[0].value = v;
                  tmp[0].index = id;
                  tmp[0].next = -1;
                }
                else if (id<tmp[tmp_start].index)
                {
                  tmp[tmp_size].value = v;
                  tmp[tmp_size].index = id;
                  tmp[tmp_size].next = tmp_start;
                  tmp_start = tmp_size;
                  tmp_el = tmp_start;
                  tmp_size++;
                }
                else
                {
                  int nextel = tmp[tmp_el].next;
                  while (nextel >= 0 && tmp[nextel].index<=id)
                  {
                    tmp_el = nextel;
                    nextel = tmp[nextel].next;
                  }

                  if (tmp[tmp_el].index==id)
                  {
                    tmp[tmp_el].value = v;
                  }
                  else
                  {
                    tmp[tmp_size].value = v;
                    tmp[tmp_size].index = id;
                    tmp[tmp_size].next = tmp[tmp_el].next;
                    tmp[tmp_el].next = tmp_size;
                    tmp_size++;
                  }
                }
              }
            }
          }
//           for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
          for (int k=0; k<j+1; ++k)
          {
//             Scalar y = m_matrix.coeff(j,k);
//             if (!ei_isMuchSmallerThan(ei_abs(y),Scalar(1)))
//             {
            int tmp_el = tmp_start;
            typename MatrixType::InnerIterator it(m_matrix, k);
            while (it && it.index()<j)
              ++it;
            if (it && it.index()==j)
            {
              Scalar y = it.value();
              x -= ei_abs2(y);
              for (; it; ++it)
              {
                Scalar v = -it.value() * y;
                int id = it.index();
                if (tmp_size==0)
                {
//                   std::cout << "insert because size==0\n";
                  tmp_start = 0;
                  tmp_el = 0;
                  tmp_size++;
                  tmp[0].value = v;
                  tmp[0].index = id;
                  tmp[0].next = -1;
                }
                else if (id<tmp[tmp_start].index)
                {
//                   std::cout << "insert because not in (0) " << id << " " << tmp[tmp_start].index << " " << tmp_start << "\n";
                  tmp[tmp_size].value = v;
                  tmp[tmp_size].index = id;
                  tmp[tmp_size].next = tmp_start;
                  tmp_start = tmp_size;
                  tmp_el = tmp_start;
                  tmp_size++;
                }
                else
                {
                  int nextel = tmp[tmp_el].next;
                  while (nextel >= 0 && tmp[nextel].index<=id)
                  {
                    tmp_el = nextel;
                    nextel = tmp[nextel].next;
                  }

                  if (tmp[tmp_el].index==id)
                  {
                    tmp[tmp_el].value -= v;
                  }
                  else
                  {
//                     std::cout << "insert because not in (1)\n";
                    tmp[tmp_size].value = v;
                    tmp[tmp_size].index = id;
                    tmp[tmp_size].next = tmp[tmp_el].next;
                    tmp[tmp_el].next = tmp_size;
                    tmp_size++;
                  }
                }
              }
            }
          }
          RealScalar rx = ei_sqrt(ei_real(x));
          m_matrix.fill(j,j) = rx;
          Scalar y = Scalar(1)/rx;
          int k = tmp_start;
          while (k>=0)
          {
            if (!ei_isMuchSmallerThan(ei_abs(tmp[k].value),Scalar(0.01)))
            {
//               std::cout << "fill " << tmp[k].index << "," << j << "\n";
              m_matrix.fill(tmp[k].index, j) = tmp[k].value * y;
            }
            k = tmp[k].next;
          }
        }
      }

    }
  }
  m_matrix.endFill();
}

#endif

/** \returns the solution of \f$ A x = b \f$ using the current decomposition of A.
  * In other words, it returns \f$ A^{-1} b \f$ computing
  * \f$ {L^{*}}^{-1} L^{-1} b \f$ from right to left.
  * \param b the column vector \f$ b \f$, which can also be a matrix.
  *
  * Example: \include Cholesky_solve.cpp
  * Output: \verbinclude Cholesky_solve.out
  *
  * \sa MatrixBase::cholesky(), CholeskyWithoutSquareRoot::solve()
  */
// template<typename MatrixType>
// template<typename Derived>
// typename Derived::Eval Cholesky<MatrixType>::solve(const MatrixBase<Derived> &b) const
// {
//   const int size = m_matrix.rows();
//   ei_assert(size==b.rows());
//
//   return m_matrix.adjoint().template part<Upper>().solveTriangular(matrixL().solveTriangular(b));
// }

#endif // EIGEN_BASICSPARSECHOLESKY_H
