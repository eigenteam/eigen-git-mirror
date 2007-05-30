/*
 * $Id: bad_xpr_const_ref.cc,v 1.1 2003/10/21 19:40:38 opetzold Exp $
 *
 * This example shows the problem on holding references
 * by expressions. On higher optimization levels all things
 * are good. Without optimizations it crashs.
 */

extern "C" int printf(const char*, ...);

#ifndef restrict
#define restrict  __restrict__
#endif

template<unsigned Sz> class Vector;

struct Fcnl_Assign {
  static inline void apply_on(double& restrict lhs, double rhs) { lhs = rhs; }
};

struct Fcnl_Add {
  static inline double apply_on(double lhs, double rhs) { return lhs + rhs; }
};

template<unsigned Sz, unsigned Stride=0>
struct MetaVector
{
  enum {
    doIt = (Stride < (Sz-1)) ? 1 : 0
  };

  template <class E1, class E2, class Fcnl>
  static inline
  void assign(E1& lhs, const E2& rhs, const Fcnl& fn) {
    fn.apply_on(lhs(Stride), rhs(Stride));
    MetaVector<Sz * doIt, (Stride+1) * doIt>::assign(lhs, rhs, fn);
  }
};

template<>
struct MetaVector<0,0>
{
  template <class E1, class E2, class Fcnl>
  static inline void assign(E1&, const E2&, const Fcnl&) { }
};


template<class E, unsigned Sz>
struct XprVector
{
  explicit XprVector(const E& e) : m_expr(e) { }

  double operator()(unsigned i) const {
    return m_expr(i);
  }

  template<class E2, class Fcnl>
  void assign_to(E2& e, const Fcnl& fn) const {
    MetaVector<Sz, 0>::assign(e, *this, fn);
  }

  const E						m_expr;
};


template<unsigned Sz, unsigned Stride=1>
struct VectorConstReference
{
  explicit VectorConstReference(const Vector<Sz>& rhs) : m_data(rhs.m_data) { }

  double operator()(unsigned i) const {
    return m_data[i * Stride];
  }

  const double* restrict 				m_data;
};


template<unsigned Sz>
struct Vector
{
  explicit Vector() { }

  double& restrict operator()(unsigned i) { return m_data[i]; }

  double operator()(unsigned i) const { return m_data[i]; }

  typedef VectorConstReference<Sz, 1>    		ConstReference;

  ConstReference const_ref() const { return ConstReference(*this); }

  template<class Fcnl>
  void assign_to(Vector& v, const Fcnl& fn) {
    MetaVector<Sz, 0>::assign(v, *this, fn);
  }

  template<class E>
  Vector& operator=(const XprVector<E, Sz>& rhs) {
    rhs.assign_to(*this, Fcnl_Assign());
    return *this;
  }

  double						m_data[Sz];
};


template<class BinOp, class E1, class E2>
struct XprBinOp
{
  explicit XprBinOp(const E1& lhs, const E2& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  { }

  double operator()(unsigned i) const {
    return BinOp::apply_on(m_lhs(i), m_rhs(i));
  }

  const E1& 						m_lhs;
  const E2& 						m_rhs;
};


template<unsigned Sz>
inline
XprVector<
  XprBinOp<
  Fcnl_Add,
    VectorConstReference<Sz>,
    VectorConstReference<Sz>
  >,
  Sz
>
add (const Vector<Sz>& lhs, const Vector<Sz>& rhs) {
  typedef XprBinOp <
    Fcnl_Add,
    VectorConstReference<Sz>,
    VectorConstReference<Sz>
  >							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


int main()
{

  Vector<5>	v, v1,v2;

  v1(0) = 1;
  v1(1) = 2;
  v1(2) = 3;
  v1(3) = 4;
  v1(4) = 5;

  v2(0) = 1;
  v2(1) = 2;
  v2(2) = 3;
  v2(3) = 4;
  v2(4) = 5;

  v = add(v1, v2);

  printf("v(0) = %f\n", v(0));
}
