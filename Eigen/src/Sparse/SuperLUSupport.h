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

#ifndef EIGEN_SUPERLUSUPPORT_H
#define EIGEN_SUPERLUSUPPORT_H

// declaration of gssvx taken from GMM++
#define DECL_GSSVX(NAMESPACE,FNAME,FLOATTYPE,KEYTYPE) \
    inline float SuperLU_gssvx(superlu_options_t *options, SuperMatrix *A,  \
         int *perm_c, int *perm_r, int *etree, char *equed,  \
         FLOATTYPE *R, FLOATTYPE *C, SuperMatrix *L,         \
         SuperMatrix *U, void *work, int lwork,              \
         SuperMatrix *B, SuperMatrix *X,                     \
         FLOATTYPE *recip_pivot_growth,                      \
         FLOATTYPE *rcond, FLOATTYPE *ferr, FLOATTYPE *berr, \
         SuperLUStat_t *stats, int *info, KEYTYPE) {         \
    NAMESPACE::mem_usage_t mem_usage;                                    \
    NAMESPACE::FNAME(options, A, perm_c, perm_r, etree, equed, R, C, L,  \
         U, work, lwork, B, X, recip_pivot_growth, rcond,    \
         ferr, berr, &mem_usage, stats, info);               \
    return mem_usage.for_lu; /* bytes used by the factor storage */     \
  }

DECL_GSSVX(SuperLU_S,sgssvx,float,float)
DECL_GSSVX(SuperLU_C,cgssvx,float,std::complex<float>)
DECL_GSSVX(SuperLU_D,dgssvx,double,double)
DECL_GSSVX(SuperLU_Z,zgssvx,double,std::complex<double>)

template<typename MatrixType>
struct SluMatrixMapHelper;

/** \internal
  *
  * A wrapper class for SuperLU matrices. It supports only compressed sparse matrices
  * and dense matrices. Supernodal and other fancy format are not supported by this wrapper.
  *
  * This wrapper class mainly aims to avoids the need of dynamic allocation of the storage structure.
  */
struct SluMatrix : SuperMatrix
{
  SluMatrix() {}

  SluMatrix(const SluMatrix& other)
    : SuperMatrix(other)
  {
    Store = &nnz;
    nnz = other.nnz;
    values = other.values;
    innerInd = other.innerInd;
    outerInd = other.outerInd;
  }

  struct
  {
    union {int nnz;int lda;};
    void *values;
    int *innerInd;
    int *outerInd;
  };

  void setStorageType(Stype_t t)
  {
    Stype = t;
    if (t==SLU_NC || t==SLU_NR || t==SLU_DN)
      Store = &nnz;
    else
    {
      ei_assert(false && "storage type not supported");
      Store = 0;
    }
  }

  template<typename Scalar>
  void setScalarType()
  {
    if (ei_is_same_type<Scalar,float>::ret)
      Dtype = SLU_S;
    else if (ei_is_same_type<Scalar,double>::ret)
      Dtype = SLU_D;
    else if (ei_is_same_type<Scalar,std::complex<float> >::ret)
      Dtype = SLU_C;
    else if (ei_is_same_type<Scalar,std::complex<double> >::ret)
      Dtype = SLU_Z;
    else
    {
      ei_assert(false && "Scalar type not supported by SuperLU");
    }
  }

  template<typename MatrixType>
  static SluMatrix Map(MatrixType& mat)
  {
    SluMatrix res;
    SluMatrixMapHelper<MatrixType>::run(mat, res);
    return res;
  }
};

template<typename Scalar, int Rows, int Cols, int StorageOrder, int MRows, int MCols>
struct SluMatrixMapHelper<Matrix<Scalar,Rows,Cols,StorageOrder,MRows,MCols> >
{
  typedef Matrix<Scalar,Rows,Cols,StorageOrder,MRows,MCols> MatrixType;
  static void run(MatrixType& mat, SluMatrix& res)
  {
    assert(StorageOrder==0 && "row-major dense matrices is not supported by SuperLU");
    res.setStorageType(SLU_DN);
    res.setScalarType<Scalar>();
    res.Mtype     = SLU_GE;

    res.nrow      = mat.rows();
    res.ncol      = mat.cols();

    res.lda       = mat.stride();
    res.values    = mat.data();
  }
};

template<typename Scalar, int Flags>
struct SluMatrixMapHelper<SparseMatrix<Scalar,Flags> >
{
  typedef SparseMatrix<Scalar,Flags> MatrixType;
  static void run(MatrixType& mat, SluMatrix& res)
  {
    if (Flags&RowMajorBit)
    {
      res.setStorageType(SLU_NR);
      res.nrow      = mat.cols();
      res.ncol      = mat.rows();
    }
    else
    {
      res.setStorageType(SLU_NC);
      res.nrow      = mat.rows();
      res.ncol      = mat.cols();
    }

    res.Mtype     = SLU_GE;

    res.nnz       = mat.nonZeros();
    res.values    = mat._valuePtr();
    res.innerInd  = mat._innerIndexPtr();
    res.outerInd  = mat._outerIndexPtr();

    res.setScalarType<Scalar>();

    // FIXME the following is not very accurate
    if (Flags & Upper)
      res.Mtype = SLU_TRU;
    if (Flags & Lower)
      res.Mtype = SLU_TRL;
    if (Flags & SelfAdjoint)
      ei_assert(false && "SelfAdjoint matrix shape not supported by SuperLU");
  }
};

template<typename Scalar, int Flags>
SluMatrix SparseMatrix<Scalar,Flags>::asSluMatrix()
{
  return SluMatrix::Map(*this);
}

template<typename Scalar, int Flags>
SparseMatrix<Scalar,Flags> SparseMatrix<Scalar,Flags>::Map(SluMatrix& sluMat)
{
  SparseMatrix res;
  if (Flags&RowMajorBit)
  {
    assert(sluMat.Stype == SLU_NR);
    res.m_innerSize   = sluMat.ncol;
    res.m_outerSize   = sluMat.nrow;
  }
  else
  {
    assert(sluMat.Stype == SLU_NC);
    res.m_innerSize   = sluMat.nrow;
    res.m_outerSize   = sluMat.ncol;
  }
  res.m_outerIndex  = sluMat.outerInd;
  SparseArray<Scalar> data = SparseArray<Scalar>::Map(
                                sluMat.innerInd,
                                reinterpret_cast<Scalar*>(sluMat.values),
                                sluMat.outerInd[res.m_outerSize]);
  res.m_data.swap(data);
  res.markAsRValue();
  return res;
}

template<typename MatrixType>
class SparseLU<MatrixType,SuperLU> : public SparseLU<MatrixType>
{
  protected:
    typedef SparseLU<MatrixType> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    using Base::m_flags;
    using Base::m_status;

  public:

    SparseLU(int flags = NaturalOrdering)
      : Base(flags)
    {
    }

    SparseLU(const MatrixType& matrix, int flags = NaturalOrdering)
      : Base(flags)
    {
      compute(matrix);
    }

    ~SparseLU()
    {
    }

    template<typename BDerived, typename XDerived>
    bool solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived>* x) const;

    void compute(const MatrixType& matrix);

  protected:
    // cached data to reduce reallocation:
    mutable SparseMatrix<Scalar> m_matrix;
    mutable SluMatrix m_sluA;
    mutable SuperMatrix m_sluL, m_sluU,;
    mutable SluMatrix m_sluB, m_sluX;
    mutable SuperLUStat_t m_sluStat;
    mutable superlu_options_t m_sluOptions;
    mutable std::vector<int> m_sluEtree, m_sluPermR, m_sluPermC;
    mutable std::vector<RealScalar> m_sluRscale, m_sluCscale;
    mutable std::vector<RealScalar> m_sluFerr, m_sluBerr;
    mutable char m_sluEqued;
};

template<typename MatrixType>
void SparseLU<MatrixType,SuperLU>::compute(const MatrixType& a)
{
  const int size = a.rows();
  m_matrix = a;

  set_default_options(&m_sluOptions);
  m_sluOptions.ColPerm = NATURAL;
  m_sluOptions.PrintStat = NO;
  m_sluOptions.ConditionNumber = NO;
  m_sluOptions.Trans = NOTRANS;

  switch (Base::orderingMethod())
  {
      case NaturalOrdering          : m_sluOptions.ColPerm = NATURAL; break;
      case MinimumDegree_AT_PLUS_A  : m_sluOptions.ColPerm = MMD_AT_PLUS_A; break;
      case MinimumDegree_ATA        : m_sluOptions.ColPerm = MMD_ATA; break;
      case ColApproxMinimumDegree   : m_sluOptions.ColPerm = COLAMD; break;
      default:
        std::cerr << "Eigen: ordering method \"" << Base::orderingMethod() << "\" not supported by the SuperLU backend\n";
        m_sluOptions.ColPerm = NATURAL;
  };

  m_sluA = m_matrix.asSluMatrix();
  memset(&m_sluL,0,sizeof m_sluL);
  memset(&m_sluU,0,sizeof m_sluU);
  m_sluEqued = 'B';
  int info = 0;

  m_sluPermR.resize(size);
  m_sluPermC.resize(size);
  m_sluRscale.resize(size);
  m_sluCscale.resize(size);
  m_sluEtree.resize(size);

  RealScalar recip_pivot_gross, rcond;
  RealScalar ferr, berr;

  // set empty B and X
  m_sluB.setStorageType(SLU_DN);
  m_sluB.setScalarType<Scalar>();
  m_sluB.Mtype = SLU_GE;
  m_sluB.values = 0;
  m_sluB.nrow = m_sluB.ncol = 0;
  m_sluB.lda = size;
  m_sluX = m_sluB;

  StatInit(&m_sluStat);
  SuperLU_gssvx(&m_sluOptions, &m_sluA, &m_sluPermC[0], &m_sluPermR[0], &m_sluEtree[0],
          &m_sluEqued, &m_sluRscale[0], &m_sluCscale[0],
          &m_sluL, &m_sluU,
          NULL, 0,
          &m_sluB, &m_sluX,
          &recip_pivot_gross, &rcond,
          &ferr, &berr,
          &m_sluStat, &info, Scalar());
  StatFree(&m_sluStat);

  // FIXME how to check for errors ???
  Base::m_succeeded = true;
}

// template<typename MatrixType>
// inline const MatrixType&
// SparseLU<MatrixType,SuperLU>::matrixL() const
// {
//   ei_assert(false && "matrixL() is Not supported by the SuperLU backend");
//   return m_matrix;
// }
//
// template<typename MatrixType>
// inline const MatrixType&
// SparseLU<MatrixType,SuperLU>::matrixU() const
// {
//   ei_assert(false && "matrixU() is Not supported by the SuperLU backend");
//   return m_matrix;
// }

template<typename MatrixType>
template<typename BDerived,typename XDerived>
bool SparseLU<MatrixType,SuperLU>::solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived> *x) const
{
  const int size = m_matrix.rows();
  const int rhsCols = b.cols();
  ei_assert(size==b.rows());

  m_sluOptions.Fact = FACTORED;
  m_sluOptions.IterRefine = NOREFINE;

  m_sluFerr.resize(rhsCols);
  m_sluBerr.resize(rhsCols);
  m_sluB = SluMatrix::Map(b.const_cast_derived());
  m_sluX = SluMatrix::Map(x->derived());

  StatInit(&m_sluStat);
  int info = 0;
  RealScalar recip_pivot_gross, rcond;
  SuperLU_gssvx(
    &m_sluOptions, &m_sluA,
    &m_sluPermC[0], &m_sluPermR[0],
    &m_sluEtree[0], &m_sluEqued,
    &m_sluRscale[0], &m_sluCscale[0],
    &m_sluL, &m_sluU,
    NULL, 0,
    &m_sluB, &m_sluX,
    &recip_pivot_gross, &rcond,
    &m_sluFerr[0], &m_sluBerr[0],
    &m_sluStat, &info, Scalar());
  StatFree(&m_sluStat);
}

#endif // EIGEN_SUPERLUSUPPORT_H
