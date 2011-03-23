
#define EIGEN_ENABLE_EVALUATORS
#include "main.h"

using internal::copy_using_evaluator;
using namespace std;

#define VERIFY_IS_APPROX_EVALUATOR(DEST,EXPR) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (EXPR).eval());
#define VERIFY_IS_APPROX_EVALUATOR2(DEST,EXPR,REF) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (REF).eval());

void test_evaluators()
{
  // Testing Matrix evaluator and Transpose
  Vector2d v = Vector2d::Random();
  const Vector2d v_const(v);
  Vector2d v2;
  RowVector2d w;

  VERIFY_IS_APPROX_EVALUATOR(v2, v);
  VERIFY_IS_APPROX_EVALUATOR(v2, v_const);

  // Testing Transpose
  VERIFY_IS_APPROX_EVALUATOR(w, v.transpose()); // Transpose as rvalue
  VERIFY_IS_APPROX_EVALUATOR(w, v_const.transpose());

  copy_using_evaluator(w.transpose(), v); // Transpose as lvalue
  VERIFY_IS_APPROX(w,v.transpose().eval());

  copy_using_evaluator(w.transpose(), v_const);
  VERIFY_IS_APPROX(w,v_const.transpose().eval());

  // Testing Array evaluator
  ArrayXXf a(2,3);
  ArrayXXf b(3,2);
  a << 1,2,3, 4,5,6;
  const ArrayXXf a_const(a);

  VERIFY_IS_APPROX_EVALUATOR(b, a.transpose());

  VERIFY_IS_APPROX_EVALUATOR(b, a_const.transpose());

  // Testing CwiseNullaryOp evaluator
  copy_using_evaluator(w, RowVector2d::Random());
  VERIFY((w.array() >= -1).all() && (w.array() <= 1).all()); // not easy to test ...

  VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Zero());

  VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Constant(3));
  
  // mix CwiseNullaryOp and transpose
  VERIFY_IS_APPROX_EVALUATOR(w, Vector2d::Zero().transpose());

  {
    int s = internal::random<int>(1,100);
    MatrixXf a(s,s), b(s,s), c(s,s), d(s,s);
    a.setRandom();
    b.setRandom();
    c.setRandom();
    d.setRandom();
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b));
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b).transpose(), (a*b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b) + prod(b,c), a*b + b*c);
    
//     copy_using_evaluator(d, a.transpose() + (a.transpose() * (b+b)));
//     cout << d << endl;
  }
  
  // this does not work because Random is eval-before-nested: 
  // copy_using_evaluator(w, Vector2d::Random().transpose());
  
  // test CwiseUnaryOp
  VERIFY_IS_APPROX_EVALUATOR(v2, 3 * v);
  VERIFY_IS_APPROX_EVALUATOR(w, (3 * v).transpose());
  VERIFY_IS_APPROX_EVALUATOR(b, (a + 3).transpose());
  VERIFY_IS_APPROX_EVALUATOR(b, (2 * a_const + 3).transpose());

  // test CwiseBinaryOp
  VERIFY_IS_APPROX_EVALUATOR(v2, v + Vector2d::Ones());
  VERIFY_IS_APPROX_EVALUATOR(w, (v + Vector2d::Ones()).transpose().cwiseProduct(RowVector2d::Constant(3)));
}
