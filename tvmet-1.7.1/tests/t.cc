#include <iostream>
#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

using std::cout;
using std::endl;

struct separator {
  std::ostream& print_on(std::ostream& os) const {
    for(std::size_t i = 0; i < 63; ++i) os << "-";
    return os;
  }
};

std::ostream& operator<<(std::ostream& os, const separator& s) {
  return s.print_on(os);
}

template<class T1, class T2 = T1>
class TestBase
{
public:
  typedef T1					value_type;
  typedef T2					value_type2;

  typedef tvmet::Vector<value_type, 3>		vector_type;
  typedef tvmet::Matrix<value_type, 3, 3>	matrix_type;
  typedef tvmet::Vector<value_type2, 3>		vector_type2;
  typedef tvmet::Matrix<value_type2, 3, 3>	matrix_type2;

private:
  vector_type					v0;
  matrix_type					M0;

protected:
  vector_type					v1, v2, v3;
  matrix_type					M1, M2, M3;

protected:
  TestBase()
  {
    v0 = 1,2,3;
    M0 = 1,4,7,2,5,8,3,6,9;
    reset();
  }

  ~TestBase() { }

  void reset()
  {
    v1 = v0; v2 = v0; v3 = v0;
    M1 = M0; M2 = M0; M3 = M0;
  }

public:
  void show_v1(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << v1 << endl
	 << separator() << endl;
  }

  void show_v2(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << v2 << endl
	 << separator() << endl;
  }

  void show_v3(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << v3 << endl
	 << separator() << endl;
  }

  void show_v() {
    cout << separator() << endl;
    cout << "v1 = " << v1 << endl
	 << "v2 = " << v2 << endl
	 << "v3 = " << v3 << endl;
    cout << separator() << endl;
  }

  void show_M1(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << M1 << endl
	 << separator() << endl;
  }

  void show_M2(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << M2 << endl
	 << separator() << endl;
  }

  void show_M3(const std::string& op) {
    cout << separator() << endl
	 << op << " = "
	 << M3 << endl
	 << separator() << endl;
  }

  void show_M() {
    cout << separator() << endl;
    cout << "M1 = " << M1 << endl
	 << "M2 = " << M2 << endl
	 << "M3 = " << M3 << endl;
    cout << separator() << endl;
  }
};




/*
 * Vector
 */
class TestV : public TestBase<double>
{
public:
  TestV() { }

public:
  void case1()  {
    reset();

    v1 = v2 + v2 + v3;

    show_v1("v2 + v2 + v3");
  }
  void case2()  {
    reset();

    v1 = sin( (v2 + v2) * v2 );

    show_v1("sin( (v2 + v2) * v2 )");
  }
  void case3()  {
    reset();

    v1 = (v2 + v2) * (v2 + v2);

    show_v1("(v2 + v2) * (v2 + v2)");
  }
  void case4()  {
    reset();

    v1 = (v2 + v2) * (v2 + v2) / 4;

    show_v1("(v2 + v2) * (v2 + v2) / 4");
  }
  void case5()  {
    reset();

  }
};

/*
 * Matrix
 */
class TestM : public TestBase<double>
{
public:
  TestM() { }

public:
  void case1()  {
    reset();

    M1 = M2 + M3;

    show_M1("M2 + M3");
  }
  void case2()  {
    reset();

    M1 = M2 + M2 + M2 + M2;

    show_M1("M2 + M2 + M2 + M2");
  }
  void case3()  {
    reset();

    /*
      XXX: missing feature element_wise XprMatrix * Xprmatrix

      M1 = ( M2 + M2 ) * ( M2 + M2 );

      M1 = tvmet::element_wise::product( M2 + M2, M2 + M2 );

      show_M1("empty");
    */
  }
  void case4()  {
    reset();

    M1 = sin(M2 + M2);			// UFUNC(XprMatrix)

    show_M1("sin(M2 + M2)");
  }
  void case5()  {
    reset();

    M1 = trans(M2);			// = XprMatrix

    show_M1("trans(M2)");
  }
  void case6()  {
    reset();

    M1 = trans(M2) + M2; 		// XprMatrix + Matrix

    show_M1("trans(M2) + M2");
  }
  void case7()  {
    reset();

    M1 = M2 + trans(M2); 		// Matrix + XprMatrix

    show_M1("M2 + trans(M2)");
  }
  void case8()  {
    reset();

    /*
     * WRONG results, should be:
     *   120  264  408
     *   144  324  504
     *   168  384  600
     * there seems to be a side effect!!
     */

    M1 = prod((M2 + M2), (M2 + M2)); 	// XprMatrix * XprMatrix

    show_M1("prod((M2 + M2), (M2 + M2))");
  }
  void case9()  {
    reset();

    M1 = (M2 + M2) * (M2 + M2); 	// XprMatrix * XprMatrix

    show_M1("(M2 + M2) * (M2 + M2)");
  }
  void case10()  {
    reset();

  }
};

/*
 * Matrix-Vector
 */
class TestMV : public TestBase<double>
{
public:
  TestMV() { }

public:
  void case1()  {
    reset();

    v1 = M1 * v2;

    show_v1("M1 * v2");
  }
  void case2()  {
    reset();

    v1 = (M1 * v2) + v2;

    show_v1("(M1 * v2) + v2");
  }
  void case3()  {
    reset();

    v1 = (M1 * v2) + (M1 * v2);

    show_v1("(M1 * v2) + (M1 * v2)");
  }
  void case4()  {
    reset();

    v1 = (M1 * v2) * (M1 * v2); 	// element wise: XprVector * XprVector

    show_v1("element_wise: (M1 * v2) * (M1 * v2)");
  }
  void case5()  {
    reset();

    using namespace tvmet::element_wise;
    v1 = (M1 * v2) / (M1 * v2); 	// element_wise: XprVector / XprVector

    show_v1("element_wise: (M1 * v2) / (M1 * v2)");
  }
  void case6()  {
    reset();

    v1 = prod(M1, v2);

    show_v1("trans_prod(M1, v2)");
  }
  void case7()  {
    reset();

    v1 = prod(M1, v2) + v2;// XprVector + Vector

    show_v1("prod(M1, v2) + v2");
  }
  void case8()  {
    reset();

    using namespace tvmet::element_wise;
    v1 += prod(M1, v2) / v2;// element_wise: XprVector + Vector

    show_v1("v1 += prod(M1, v2) / v2");
  }
  void case9()  {
    reset();

    v1 = prod(M1, v2) + prod(M1, v2);// element wise: XprVector * XprVector

    show_v1("prod(M1, v2) + prod(M1, v2)");
  }
  void case10()  {
    reset();

    using namespace tvmet::element_wise;
    v1 = prod(M1, v2) / prod(M1, v2);// element_wise: XprVector / XprVector

    all_elements( v1 == 1 );

    show_v1("prod(M1, v2) / prod(M1, v2)");
  }
  void case11()  {
    reset();

    v1 = M1 * (v1+v1);

    show_v1("M1 * (v1+v1)");
  }
  void case12()  {
    reset();

    v1 = M1 * prod(M1, v2);

    show_v1("M1 * prod(M1, v2)");
  }
};





/*
 * Main
 */
int main()
{
  tvmet::Matrix<double, 3,3> MM;
  MM = 1,2,3,4,5,6,7,8,9;
  tvmet::Matrix<double, 3,3> MM2( MM );

  TestV						v;
  TestM						M;
  TestMV					Mv;

  v.show_v();
  M.show_M();

  cout << "*****************************************************************" << endl;

#if 1
  v.case1();
  v.case2();
  v.case3();
  v.case4();
  v.case5();

  cout << "*****************************************************************" << endl;
#endif

#if 1
  M.case1();
  M.case2();
  M.case3();
  M.case4();
  M.case5();
  M.case6();
  M.case7();
  M.case8();
  M.case9();
  M.case10();

  cout << "*****************************************************************" << endl;
#endif

#if 1
  Mv.case1();
  Mv.case2();
  Mv.case3();
  Mv.case4();
  Mv.case5();
#endif
  Mv.case6();
  Mv.case7();
  Mv.case8();
  Mv.case9();
  Mv.case10();
  Mv.case11();
  Mv.case12();
}
