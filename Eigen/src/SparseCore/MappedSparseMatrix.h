// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MAPPED_SPARSEMATRIX_H
#define EIGEN_MAPPED_SPARSEMATRIX_H

namespace Eigen {

/** \deprecated Use Map<SparseMatrix<> >
  * \class MappedSparseMatrix
  *
  * \brief Sparse matrix
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  *
  * See http://www.netlib.org/linalg/html_templates/node91.html for details on the storage scheme.
  *
  */
namespace internal {
template<typename _Scalar, int _Flags, typename _Index>
struct traits<MappedSparseMatrix<_Scalar, _Flags, _Index> > : traits<SparseMatrix<_Scalar, _Flags, _Index> >
{};
} // end namespace internal

template<typename _Scalar, int _Flags, typename _Index>
class MappedSparseMatrix
  : public Map<SparseMatrix<_Scalar, _Flags, _Index> >
{
    typedef Map<SparseMatrix<_Scalar, _Flags, _Index> > Base;

  public:
    
    typedef typename Base::Index Index;
    typedef typename Base::Scalar Scalar;

    inline MappedSparseMatrix(Index rows, Index cols, Index nnz, Index* outerIndexPtr, Index* innerIndexPtr, Scalar* valuePtr, Index* innerNonZeroPtr = 0)
      : Base(rows, cols, nnz, outerIndexPtr, innerIndexPtr, valuePtr, innerNonZeroPtr)
    {}

    /** Empty destructor */
    inline ~MappedSparseMatrix() {}
};

namespace internal {

template<typename _Scalar, int _Options, typename _Index>
struct evaluator<MappedSparseMatrix<_Scalar,_Options,_Index> >
  : evaluator<SparseCompressedBase<MappedSparseMatrix<_Scalar,_Options,_Index> > >
{
  typedef MappedSparseMatrix<_Scalar,_Options,_Index> XprType;
  typedef evaluator<SparseCompressedBase<XprType> > Base;
  
  evaluator() : Base() {}
  explicit evaluator(const XprType &mat) : Base(mat) {}
};

}

} // end namespace Eigen

#endif // EIGEN_MAPPED_SPARSEMATRIX_H
