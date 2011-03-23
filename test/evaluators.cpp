
#define EIGEN_ENABLE_EVALUATORS
#include "main.h"

using internal::copy_using_evaluator;
using namespace std;

void test_evaluators()
{
  // Testing Matrix evaluator and Transpose
  Vector2d v(1,2);
  const Vector2d v_const(v);
  Vector2d v2;
  RowVector2d w;

  copy_using_evaluator(v2, v);
  assert(v2.isApprox((Vector2d() << 1,2).finished()));

  copy_using_evaluator(v2, v_const);
  assert(v2.isApprox((Vector2d() << 1,2).finished()));

  // Testing Transpose
  copy_using_evaluator(w, v.transpose()); // Transpose as rvalue
  assert(w.isApprox((RowVector2d() << 1,2).finished()));

  copy_using_evaluator(w, v_const.transpose());
  assert(w.isApprox((RowVector2d() << 1,2).finished()));

  copy_using_evaluator(w.transpose(), v); // Transpose as lvalue
  assert(w.isApprox((RowVector2d() << 1,2).finished()));

  copy_using_evaluator(w.transpose(), v_const);
  assert(w.isApprox((RowVector2d() << 1,2).finished()));

  // Testing Array evaluator
  ArrayXXf a(2,3);
  ArrayXXf b(3,2);
  a << 1,2,3, 4,5,6;
  const ArrayXXf a_const(a);

  ArrayXXf b_expected(3,2);
  b_expected << 1,4, 2,5, 3,6;
  copy_using_evaluator(b, a.transpose());
  assert(b.isApprox(b_expected));

  copy_using_evaluator(b, a_const.transpose());
  assert(b.isApprox(b_expected));

  // Testing CwiseNullaryOp evaluator
  copy_using_evaluator(w, RowVector2d::Random());
  assert((w.array() >= -1).all() && (w.array() <= 1).all()); // not easy to test ...

  copy_using_evaluator(w, RowVector2d::Zero());
  assert(w.isApprox((RowVector2d() << 0,0).finished()));

  copy_using_evaluator(w, RowVector2d::Constant(3));
  assert(w.isApprox((RowVector2d() << 3,3).finished()));
  
  // mix CwiseNullaryOp and transpose
  copy_using_evaluator(w, Vector2d::Zero().transpose());
  assert(w.isApprox((RowVector2d() << 0,0).finished()));

  {
    MatrixXf a(2,2), b(2,2), c(2,2), d(2,2);
    a << 1, 2, 3, 4; b << 5, 6, 7, 8; c << 9, 10, 11, 12;
    copy_using_evaluator(d, (a + b));
    cout << d << endl;
    
    copy_using_evaluator(d, (a + b).transpose());
    cout << d << endl;
    
    copy_using_evaluator(d, prod(a,b).transpose());
    cout << d << endl;
    
//     copy_using_evaluator(d, a.transpose() + (a.transpose() * (b+b)));
//     cout << d << endl;
  }
  
  // this does not work because Random is eval-before-nested: 
  // copy_using_evaluator(w, Vector2d::Random().transpose());
  
  // test CwiseUnaryOp
  copy_using_evaluator(v2, 3 * v);
  assert(v2.isApprox((Vector2d() << 3,6).finished()));

  copy_using_evaluator(w, (3 * v).transpose());
  assert(w.isApprox((RowVector2d() << 3,6).finished()));

  copy_using_evaluator(b, (a + 3).transpose());
  b_expected << 4,7, 5,8, 6,9;
  assert(b.isApprox(b_expected));

  copy_using_evaluator(b, (2 * a_const + 3).transpose());
  b_expected << 5,11, 7,13, 9,15;
  assert(b.isApprox(b_expected));

  // test CwiseBinaryOp
  copy_using_evaluator(v2, v + Vector2d::Ones());
  assert(v2.isApprox((Vector2d() << 2,3).finished()));

  copy_using_evaluator(w, (v + Vector2d::Ones()).transpose().cwiseProduct(RowVector2d::Constant(3)));
  assert(w.isApprox((RowVector2d() << 6,9).finished()));
}
