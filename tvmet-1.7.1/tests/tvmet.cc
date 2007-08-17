/*
 * $Id: tvmet.cc,v 1.3 2003/10/21 19:37:06 opetzold Exp $
 *
 * This file shows the basic principle used by tvmet. Therefore
 * you will not find promotion etc. here.
 */

extern "C" int printf(const char*, ...);

#ifndef restrict
#define restrict  __restrict__
#endif

template<unsigned Rows, unsigned Cols> class Matrix;

struct XprNull { explicit XprNull() { } };

static inline
double operator+(const double& lhs, XprNull) { return lhs; }


struct Fcnl_Assign { static inline void apply_on(double& restrict lhs, double rhs) { lhs = rhs; } };


template<unsigned Rows, unsigned Cols,
	 unsigned RowStride, unsigned ColStride>
struct MetaMatrix
{
  enum {
    doRows = (RowStride < Rows - 1) ? 1 : 0,
    doCols = (ColStride < Cols - 1) ? 1 : 0
  };

  template<class E1, class E2, class Fcnl>
  static inline
  void assign2(E1& lhs, const E2& rhs, const Fcnl& fn) {
    fn.apply_on( lhs(RowStride, ColStride), rhs(RowStride, ColStride) );
    MetaMatrix<Rows * doCols, Cols * doCols, RowStride * doCols, (ColStride+1) * doCols>::assign2(lhs, rhs, fn);
  }

  template<class E1, class E2, class Fcnl>
  static inline
  void assign(E1& lhs, const E2& rhs, const Fcnl& fn) {
    MetaMatrix<Rows, Cols, RowStride, 0>::assign2(lhs, rhs, fn);
    MetaMatrix<Rows * doRows, Cols * doRows, (RowStride+1) * doRows, 0>::assign(lhs, rhs, fn);
  }
};

template<>
struct MetaMatrix<0, 0, 0, 0>
{
  template<class E1, class E2, class Fcnl>
  static inline void assign2(E1&, const E2&, const Fcnl&) { }

  template<class E1, class E2, class Fcnl>
  static inline void assign(E1&, const E2&, const Fcnl&) { }
};


template<unsigned Rows1, unsigned Cols1,
	 unsigned Cols2,
	 unsigned RowStride1, unsigned ColStride1,
	 unsigned RowStride2, unsigned ColStride2,
	 unsigned K>
struct MetaGemm
{
  enum { doIt = (K != Cols1 - 1) };

  template<class E1, class E2>
  static inline
  double prod(const E1& lhs, const E2& rhs, unsigned i, unsigned j) {
    return lhs(i, K) * rhs(K, j)
      + MetaGemm<Rows1 * doIt, Cols1 * doIt,
      Cols2 * doIt, RowStride1 * doIt, ColStride1 * doIt,
      RowStride2 * doIt, ColStride2 * doIt, (K+1) * doIt>::prod(lhs, rhs, i, j);
  }
};

template<>
struct MetaGemm<0,0,0,0,0,0,0,0>
{
  template<class E1, class E2>
  static inline XprNull prod(const E1&, const E2&, unsigned, unsigned) { return XprNull(); }
};


template<class E1, class E2,
         unsigned Rows1, unsigned Cols1,
	 unsigned Cols2,
	 unsigned RowStride1, unsigned ColStride1,
	 unsigned RowStride2, unsigned ColStride2>
struct XprMMProduct
{
  explicit XprMMProduct(const E1& lhs, const E2& rhs) : m_lhs(lhs), m_rhs(rhs) { }

  double operator()(unsigned i, unsigned j) const {
    return MetaGemm<
      Rows1, Cols1,
      Cols2,
      RowStride1, ColStride1,
      RowStride2, ColStride2, 0>::prod(m_lhs, m_rhs, i, j);
  }

//   void assign_to(Matrix<Rows1, Cols2>& rhs) const {
//     MetaMatrix<Rows1, Cols2, 0, 0>::assign(rhs, *this, Fcnl_Assign());
//   }

private:
  const E1					m_lhs;
  const E2		 			m_rhs;
};


template<class E>
struct XprMatrixTranspose
{
  explicit XprMatrixTranspose(const E& e) : m_expr(e) { }

  double operator()(unsigned i, unsigned j) const { return m_expr(j, i); }

//   template<unsigned Rows, unsigned Cols>
//   void assign_to(Matrix<Rows, Cols>& rhs) const {
//     MetaMatrix<Rows, Cols, 0, 0>::assign(rhs, *this, Fcnl_Assign());
//   }

private:
  const E					m_expr;
};


template<class E, unsigned Rows, unsigned Cols>
struct XprMatrix
{
  explicit XprMatrix(const E& e) : m_expr(e) { }

  double operator()(unsigned i, unsigned j) const { return m_expr(i, j); }

  void assign_to(Matrix<Rows, Cols>& rhs) const {
    MetaMatrix<Rows, Cols, 0, 0>::assign(rhs, *this, Fcnl_Assign());
  }

private:
  const E					m_expr;
};


template<unsigned Rows, unsigned Cols,
	 unsigned RowStride, unsigned ColStride>
struct MatrixConstRef
{
  explicit MatrixConstRef(const Matrix<Rows, Cols>& rhs) : m_data(rhs.m_data) { }

  double operator()(unsigned i, unsigned j) const {
    return m_data[i * RowStride + j * ColStride];
  }

private:
  const double* restrict 			m_data;
};


template<unsigned Rows, unsigned Cols>
struct Matrix
{
  explicit Matrix() { m_data = new double [Rows*Cols]; }

  template<class E>
  explicit Matrix(const XprMatrix<E, Rows, Cols>& rhs) {
    m_data = new double [Rows*Cols];
    MetaMatrix<Rows, Cols, 0, 0>::assign(*this, rhs, Fcnl_Assign());
  }

  ~Matrix() { delete [] m_data; }

  double& restrict operator()(unsigned i, unsigned j) { return m_data[i * Cols + j]; }

  double operator()(unsigned i, unsigned j) const { return m_data[i * Cols + j]; }

  MatrixConstRef<Rows,Cols,Cols,1> constRef() const {
    return MatrixConstRef<Rows,Cols,Cols,1>(*this);
  }

  Matrix& operator=(const Matrix<Rows, Cols>& rhs) {
    rhs.assign_to(*this);
    return *this;
  }

  void assign_to(Matrix<Rows, Cols>& rhs) const {
    MetaMatrix<Rows, Cols, 0, 0>::assign(rhs, *this, Fcnl_Assign());
  }

  template <class E>
  Matrix& operator=(const XprMatrix<E, Rows, Cols>& rhs) {
    rhs.assign_to(*this);
    return *this;
  }

  template <class E>
  void assign_to(XprMatrix<E, Rows, Cols>& rhs) const {
    MetaMatrix<Rows, Cols, 0, 0>::assign(rhs, *this, Fcnl_Assign());
  }

  void print() const {
    printf("[\n");
    for(unsigned i = 0; i != Rows; ++i) {
      printf("\t[");
      for(unsigned j = 0; j != Cols; ++j)
	printf("\t%+4.2f", this->operator()(i, j));
      printf("]\n");
    }
    printf("]\n");
  }

  double* 						m_data;
};


template<unsigned Rows1, unsigned Cols1,
	 unsigned Cols2>
inline
XprMatrix<
  XprMMProduct<
    MatrixConstRef<Rows1, Cols1, Cols1, 1>,
    MatrixConstRef<Cols1, Cols2, Cols2, 1>,
    Rows1, Cols1,	// M1(Rows1, Cols1)
    Cols2, 		// M2(Cols1, Cols2)
    Cols1, 1, 		// Stride M1
    Cols2, 1		// Stride M2
  >,
  Rows1, Cols2		// return Dim
>
prod(const Matrix<Rows1, Cols1>& lhs, const Matrix<Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    MatrixConstRef<Rows1, Cols1, Cols1, 1>,
    MatrixConstRef<Cols1, Cols2, Cols2, 1>,
    Rows1, Cols1,
    Cols2,
    Cols1, 1,
    Cols2, 1
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(
    expr_type(lhs.constRef(), rhs.constRef()));
}

template<class E1, unsigned Rows1, unsigned Cols1, unsigned Cols2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>,
    MatrixConstRef<Cols1, Cols2, Cols2, 1>,
    Rows1, Cols1, Cols2,
    Cols1, 1, Cols2, 1
  >,
  Rows1, Cols2
  >
prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>,
    MatrixConstRef<Cols1, Cols2, Cols2, 1>,
    Rows1, Cols1, Cols2,
    Cols1, 1, Cols2, 1
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(expr_type(lhs, rhs.constRef()));
}


template<unsigned Rows, unsigned Cols>
inline
XprMatrix<
  XprMatrixTranspose<
    MatrixConstRef<Rows, Cols, Cols, 1>
  >,
  Cols, Rows
>
trans(const Matrix<Rows, Cols>& rhs) {
  typedef XprMatrixTranspose<
    MatrixConstRef<Rows, Cols, Cols, 1>
  >							expr_type;
  return XprMatrix<expr_type, Cols, Rows>(expr_type(rhs.constRef()));
}


/**
 * Test driver
 */
int main()
{
  Matrix<3,2>		B;
  Matrix<3,3>		D;

  B(0,0) = -0.05;	B(0,1) =  0;
  B(1,0) =  0;		B(1,1) =  0.05;
  B(2,0) =  0.05;	B(2,1) = -0.05;

  D(0,0) = 2000;	D(0,1) = 1000;		D(0,2) = 0;
  D(1,0) = 1000;	D(1,1) = 2000;		D(1,2) = 0;
  D(2,0) = 0;		D(2,1) = 0;		D(2,2) = 500;

  printf("B = ");
  B.print();
  printf("D = ");
  D.print();
  printf("\n***********************************************\n");

  Matrix<2,2>		K;

  K = prod(prod(trans(B), D), B);

  printf("Check: (equal prod(prod(trans(B), D), B)\n");
  printf(" K = ");
  K.print();
}
