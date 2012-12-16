#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 1000000
#endif

enum func_opt
{
    TV,
    TMATRIXV
};


template <class res, class arg1, class arg2, int opt>
struct func;

template <class res, class arg1, class arg2>
struct func<res, arg1, arg2, TV>
{
    static __attribute__ ((noinline)) res run( arg1& a1, arg2& a2 )
    {
	asm ("");
	return a1 * a2;
    }
};

template <class res, class arg1, class arg2>
struct func<res, arg1, arg2, TMATRIXV>
{
    static __attribute__ ((noinline)) res run( arg1& a1, arg2& a2 )
    {
	asm ("");
	return a1.matrix() * a2;
    }
};


template <class func, class arg1, class arg2>
struct test_transform
{
    static void run()
    {
	arg1 a1;
	a1.setIdentity();
	arg2 a2;
	a2.setIdentity();

	BenchTimer timer;
	timer.reset();
	for (int k=0; k<10; ++k)
	{
	    timer.start();
	    for (int k=0; k<REPEAT; ++k)
		a2 = func::run( a1, a2 );
	    timer.stop();
	}
	std::cout << timer.value() << "s  " << (double(REPEAT)/timer.value())/(1024.*1024.*1024.) << " GFlops\n";
    }
};


#define run_test( op, scalar, mode, option, vsize ) \
    std::cout << #op << " " << #scalar << " " << #mode << " " << #option << " " << #vsize " "; \
    {\
	typedef Transform<scalar, 3, mode, option> Trans;\
	typedef Matrix<scalar, vsize, 1, option> Vec;\
	typedef func<Vec,Trans,Vec,op> Func;\
	test_transform< Func, Trans, Vec >::run();\
    }

int main(int argc, char* argv[])
{
    run_test(TV, float, Isometry, AutoAlign, 3);
    run_test(TV, float, Isometry, DontAlign, 3);
    run_test(TV, float, Isometry, AutoAlign, 4);
    run_test(TV, float, Isometry, DontAlign, 4);

    run_test(TMATRIXV, float, Isometry, AutoAlign, 4);
    run_test(TMATRIXV, float, Isometry, DontAlign, 4);
}
