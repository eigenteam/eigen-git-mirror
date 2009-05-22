// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_SPARSESETTER_H
#define EIGEN_SPARSESETTER_H

template<typename MatrixType, int AccessPattern,
  int IsSupported = ei_support_access_pattern<MatrixType,AccessPattern>::ret>
struct ei_sparse_setter_selector;

/** \class SparseSetter
  *
  * Goal: provides a unified API to fill/update a dense or sparse matrix.
  *
  * Usage:
  * \code
  * {
  *   SparseSetter<MatrixType, RandomAccessPattern> w(m);
  *   for (...) w->coeffRef(rand(),rand()) = rand();
  * }
  * \endcode
  *
  * In the above example we want to fill a matrix m (could be a SparseMatrix or whatever other matrix type)
  * in a random fashion (whence the RandomAccessPattern). Internally, if \a MatrixType supports random writes
  * then \c w behaves as a pointer to m, and m is filled directly. Otherwise, a temporary matrix supporting
  * random writes is created and \c w behaves as a pointer to this temporary object. When the object \c w
  * is deleted (at the end of the block), then the temporary object is assigned to the matrix m.
  * 
  * So far we can distinghished 4 types of access pattern:
  * - FullyCoherentAccessPattern (if col major, i+j*rows must increase)
  * - InnerCoherentAccessPattern (if col major, i must increase for each column j)
  * - OuterCoherentAccessPattern (if col major, the column j is set in a random order, but j must increase)
  * - RandomAccessPattern
  * 
  * See the wiki for more details.
  * 
  * The template class ei_support_access_pattern is used to determine the type of the temporary object (which
  * can be a reference to \a MatrixType if \a MatrixType support \a AccessPattern)
  *
  * Currently only the RandomAccessPattern seems to work as expected.
  *
  * \todo define the API for each kind of access pattern
  * \todo allows both update and set modes (set start a new matrix)
  * \todo implement the OuterCoherentAccessPattern
  *
  */
template<typename MatrixType,
         int AccessPattern,
         typename WrapperType = typename ei_sparse_setter_selector<MatrixType,AccessPattern>::type>
class SparseSetter
{
    typedef typename ei_unref<WrapperType>::type _WrapperType;
  public:

    inline SparseSetter(MatrixType& matrix) : m_wrapper(matrix), mp_matrix(&matrix) {}

    ~SparseSetter()
    { *mp_matrix = m_wrapper; }

    inline _WrapperType* operator->() { return &m_wrapper; }

    inline _WrapperType& operator*() { return m_wrapper; }

  protected:

    WrapperType m_wrapper;
    MatrixType* mp_matrix;
};

template<typename MatrixType, int AccessPattern>
struct ei_sparse_setter_selector<MatrixType, AccessPattern, AccessPatternSupported>
{
  typedef MatrixType& type;
};

// forward each derived of SparseMatrixBase to the generic SparseMatrixBase specializations
template<typename Scalar, int Flags, int AccessPattern>
struct ei_sparse_setter_selector<SparseMatrix<Scalar,Flags>, AccessPattern, AccessPatternNotSupported>
: public ei_sparse_setter_selector<SparseMatrixBase<SparseMatrix<Scalar,Flags> >,AccessPattern, AccessPatternNotSupported>
{};

template<typename Scalar, int Flags, int AccessPattern>
struct ei_sparse_setter_selector<LinkedVectorMatrix<Scalar,Flags>, AccessPattern, AccessPatternNotSupported>
: public ei_sparse_setter_selector<LinkedVectorMatrix<SparseMatrix<Scalar,Flags> >,AccessPattern, AccessPatternNotSupported>
{};

template<typename Scalar, int Flags, int AccessPattern>
struct ei_sparse_setter_selector<HashMatrix<Scalar,Flags>, AccessPattern, AccessPatternNotSupported>
: public ei_sparse_setter_selector<HashMatrix<SparseMatrix<Scalar,Flags> >,AccessPattern, AccessPatternNotSupported>
{};

// generic SparseMatrixBase specializations
template<typename Derived>
struct ei_sparse_setter_selector<SparseMatrixBase<Derived>, RandomAccessPattern, AccessPatternNotSupported>
{
  typedef HashMatrix<typename Derived::Scalar, Derived::Flags> type;
};

template<typename Derived>
struct ei_sparse_setter_selector<SparseMatrixBase<Derived>, OuterCoherentAccessPattern, AccessPatternNotSupported>
{
  typedef HashMatrix<typename Derived::Scalar, Derived::Flags> type;
};

template<typename Derived>
struct ei_sparse_setter_selector<SparseMatrixBase<Derived>, InnerCoherentAccessPattern, AccessPatternNotSupported>
{
  typedef LinkedVectorMatrix<typename Derived::Scalar, Derived::Flags> type;
};

template<typename Derived>
struct ei_sparse_setter_selector<SparseMatrixBase<Derived>, FullyCoherentAccessPattern, AccessPatternNotSupported>
{
  typedef SparseMatrix<typename Derived::Scalar, Derived::Flags> type;
};

#endif // EIGEN_SPARSESETTER_H
