/*
 * This acts as an example on how to not introduce temporaries
 * for evaluating expressions. The problem is related to the
 * prod() function, where the temps are going out of scope.
 */
extern "C" int printf(const char*, ...);

#ifndef restrict
#define restrict  __restrict__
#endif

template<unsigned Rows, unsigned Cols> class Matrix;

struct XprNull { explicit XprNull() { } };

static inline
double operator+(const double& lhs, XprNull) { return lhs; }


struct fcnl_Assign { static inline void applyOn(double& restrict lhs, double rhs) { lhs = rhs; } };


template<unsigned Rows, unsigned Cols,
	 unsigned RowStride, unsigned ColStride>
struct MetaMatrix
{
  enum {
    doRows = (RowStride < Rows - 1) ? 1 : 0,
    doCols = (ColStride < Cols - 1) ? 1 : 0
  };

  template<class Mtrx, class E, class Fcnl>
  static inline
  void assign2(Mtrx& mat, const E& expr, const Fcnl& fn) {
    fn.applyOn(mat(RowStride, ColStride), expr(RowStride, ColStride));
    MetaMatrix<Rows * doCols, Cols * doCols, RowStride * doCols, (ColStride+1) * doCols>::assign2(mat, expr, fn);
  }

  template<class Mtrx, class E, class Fcnl>
  static inline
  void assign(Mtrx& mat, const E& expr, const Fcnl& fn) {
    MetaMatrix<Rows, Cols, RowStride, 0>::assign2(mat, expr, fn);
    MetaMatrix<Rows * doRows, Cols * doRows, (RowStride+1) * doRows, 0>::assign(mat, expr, fn);
  }
};

template<>
struct MetaMatrix<0, 0, 0, 0>
{
  template<class Mtrx, class E, class Fcnl>
  static inline void assign2(Mtrx&, const E&, const Fcnl&) { }

  template<class Mtrx, class E, class Fcnl>
  static inline void assign(Mtrx&, const E&, const Fcnl&) { }
};


template<unsigned Rows1, unsigned Cols1,
	 unsigned Cols2,
	 unsigned RowStride1, unsigned ColStride1,
	 unsigned RowStride2, unsigned ColStride2,
	 unsigned K>
struct MetaGemm
{
  enum { doIt = (K != Cols1 - 1) };

  static inline
  double prod(const double* restrict lhs, const double* restrict rhs, unsigned i, unsigned j) {
    return lhs[i * RowStride1 + K * ColStride1] * rhs[K * RowStride2 + j * ColStride2]
      + MetaGemm<Rows1 * doIt, Cols1 * doIt,
                 Cols2 * doIt, RowStride1 * doIt, ColStride1 * doIt,
                 RowStride2 * doIt, ColStride2 * doIt, (K+1) * doIt>::prod(lhs, rhs, i, j);
  }
};

template<>
struct MetaGemm<0,0,0,0,0,0,0,0>
{
  static inline XprNull prod(const void*, const void*, unsigned, unsigned) { return XprNull(); }
};


template<unsigned Rows1, unsigned Cols1,
	 unsigned Cols2,
	 unsigned RowStride1, unsigned ColStride1,
	 unsigned RowStride2, unsigned ColStride2>
struct XprMMProduct
{
  explicit XprMMProduct(const double* restrict lhs, const double* restrict rhs) : m_lhs(lhs), m_rhs(rhs) { }

  double operator()(unsigned i, unsigned j) const {
    return MetaGemm<Rows1, Cols1,
                    Cols2,
                    RowStride1, ColStride1,
                    RowStride2, ColStride2, 0>::prod(m_lhs, m_rhs, i, j);
  }

private:
  const double* restrict 			m_lhs;
  const double* restrict 			m_rhs;
};


template<class E>
struct XprMatrixTranspose
{
  explicit XprMatrixTranspose(const E& e) : m_expr(e) { }

  double operator()(unsigned i, unsigned j) const { return m_expr(j, i); }

private:
  const E& restrict				m_expr;
};


template<class E, unsigned Rows, unsigned Cols>
struct XprMatrix
{
  explicit XprMatrix(const E& e) : m_expr(e) { }

  double operator()(unsigned i, unsigned j) const { return m_expr(i, j); }

private:
  const E& restrict				m_expr;
};


template<unsigned Rows, unsigned Cols,
	 unsigned RowStride, unsigned ColStride>
struct MatrixConstReference
{
  explicit MatrixConstReference(const Matrix<Rows, Cols>& rhs) : m_data(rhs.m_data) { }

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
    MetaMatrix<Rows, Cols, 0, 0>::assign(*this, rhs, fcnl_Assign());
  }

  ~Matrix() { delete [] m_data; }

  double& restrict operator()(unsigned i, unsigned j) { return m_data[i * Cols + j]; }

  double operator()(unsigned i, unsigned j) const { return m_data[i * Cols + j]; }

  MatrixConstReference<Rows,Cols,Cols,1> const_ref() const {
    return MatrixConstReference<Rows,Cols,Cols,1>(*this);
  }

  template <class E> Matrix& operator=(const XprMatrix<E, Rows, Cols>& rhs) {
    MetaMatrix<Rows, Cols, 0, 0>::assign(*this, rhs, fcnl_Assign());
    return *this;
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

// private:
  double* 						m_data;
};


template<class E1, unsigned Rows1, unsigned Cols1, unsigned Cols2>
inline
XprMatrix<
  XprMMProduct<
    Rows1, Cols1, Cols2,
    Cols1, 1, Cols2, 1
  >,
  Rows1, Cols2
>
prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    Rows1, Cols1, Cols2,
    Cols1, 1, Cols2, 1
  >							expr_type;
  Matrix<Rows1, Cols1> 					temp_lhs(lhs);

  return XprMatrix<expr_type, Rows1, Cols2>(expr_type(temp_lhs.m_data, rhs.m_data));
}

template<unsigned Rows, unsigned Cols>
inline
XprMatrix<
  XprMatrixTranspose<
    MatrixConstReference<Rows, Cols, Cols, 1>
  >,
  Cols, Rows
>
trans(const Matrix<Rows, Cols>& rhs) {
  typedef XprMatrixTranspose<
    MatrixConstReference<Rows, Cols, Cols, 1>
  >							expr_type;
  return XprMatrix<expr_type, Cols, Rows>(expr_type(rhs.const_ref()));
}


/**
 * Test driver
 */
using namespace std;

int main()
{
  Matrix<3,2>		B;
  Matrix<3,3>		D;
  Matrix<2,2>		K;

  B(0,0) = -0.05;	B(0,1) =  0;
  B(1,0) =  0;		B(1,1) =  0.05;
  B(2,0) =  0.05;	B(2,1) = -0.05;

  D(0,0) = 2000;	D(0,1) = 1000;		D(0,2) = 0;
  D(1,0) = 1000;	D(1,1) = 2000;		D(1,2) = 0;
  D(2,0) = 0;		D(2,1) = 0;		D(2,2) = 500;

  K = prod(prod(trans(B), D), B);

  printf("K = ");
  K.print();	// wrong result, should be symetric
}
