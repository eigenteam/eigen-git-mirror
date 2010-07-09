
#include <iostream>
#include <Eigen/Core>
#include <bench/BenchTimer.h>
using namespace Eigen;

#ifndef SIZE
#define SIZE 50
#endif

#ifndef REPEAT
#define REPEAT 10000
#endif

typedef float Scalar;

__attribute__ ((noinline)) void benchVec(Scalar* a, Scalar* b, Scalar* c, int size);
__attribute__ ((noinline)) void benchVec(MatrixXf& a, MatrixXf& b, MatrixXf& c);
__attribute__ ((noinline)) void benchVec(VectorXf& a, VectorXf& b, VectorXf& c);

int main(int argc, char* argv[])
{
    int size = SIZE * 8;
    int size2 = size * size;
    Scalar* a = ei_aligned_new<Scalar>(size2);
    Scalar* b = ei_aligned_new<Scalar>(size2+4)+1;
    Scalar* c = ei_aligned_new<Scalar>(size2); 
    
    for (int i=0; i<size; ++i)
    {
        a[i] = b[i] = c[i] = 0;
    }
    
    BenchTimer timer;
    
    timer.reset();
    for (int k=0; k<10; ++k)
    {
        timer.start();
        benchVec(a, b, c, size2);
        timer.stop();
    }
    std::cout << timer.value() << "s  " << (double(size2*REPEAT)/timer.value())/(1024.*1024.*1024.) << " GFlops\n";
    return 0;
    for (int innersize = size; innersize>2 ; --innersize)
    {
        if (size2%innersize==0)
        {
            int outersize = size2/innersize;
            MatrixXf ma = Map<MatrixXf>(a, innersize, outersize );
            MatrixXf mb = Map<MatrixXf>(b, innersize, outersize );
            MatrixXf mc = Map<MatrixXf>(c, innersize, outersize );
            timer.reset();
            for (int k=0; k<3; ++k)
            {
                timer.start();
                benchVec(ma, mb, mc);
                timer.stop();
            }
            std::cout << innersize << " x " << outersize << "  " << timer.value() << "s   " << (double(size2*REPEAT)/timer.value())/(1024.*1024.*1024.) << " GFlops\n";
        }
    }
    
    VectorXf va = Map<VectorXf>(a, size2);
    VectorXf vb = Map<VectorXf>(b, size2);
    VectorXf vc = Map<VectorXf>(c, size2);
    timer.reset();
    for (int k=0; k<3; ++k)
    {
        timer.start();
        benchVec(va, vb, vc);
        timer.stop();
    }
    std::cout << timer.value() << "s   " << (double(size2*REPEAT)/timer.value())/(1024.*1024.*1024.) << " GFlops\n";

    return 0;
}

void benchVec(MatrixXf& a, MatrixXf& b, MatrixXf& c)
{
    for (int k=0; k<REPEAT; ++k)
        a = a + b;
}

void benchVec(VectorXf& a, VectorXf& b, VectorXf& c)
{
    for (int k=0; k<REPEAT; ++k)
        a = a + b;
}

void benchVec(Scalar* a, Scalar* b, Scalar* c, int size)
{
    typedef ei_packet_traits<Scalar>::type PacketScalar;
    const int PacketSize = ei_packet_traits<Scalar>::size;
    PacketScalar a0, a1, a2, a3, b0, b1, b2, b3;
    for (int k=0; k<REPEAT; ++k)
        for (int i=0; i<size; i+=PacketSize*8)
        {
//             a0 = ei_pload(&a[i]);
//             b0 = ei_pload(&b[i]);
//             a1 = ei_pload(&a[i+1*PacketSize]);
//             b1 = ei_pload(&b[i+1*PacketSize]);
//             a2 = ei_pload(&a[i+2*PacketSize]);
//             b2 = ei_pload(&b[i+2*PacketSize]);
//             a3 = ei_pload(&a[i+3*PacketSize]);
//             b3 = ei_pload(&b[i+3*PacketSize]);
//             ei_pstore(&a[i], ei_padd(a0, b0));
//             a0 = ei_pload(&a[i+4*PacketSize]);
//             b0 = ei_pload(&b[i+4*PacketSize]);
//             
//             ei_pstore(&a[i+1*PacketSize], ei_padd(a1, b1));
//             a1 = ei_pload(&a[i+5*PacketSize]);
//             b1 = ei_pload(&b[i+5*PacketSize]);
//             
//             ei_pstore(&a[i+2*PacketSize], ei_padd(a2, b2));
//             a2 = ei_pload(&a[i+6*PacketSize]);
//             b2 = ei_pload(&b[i+6*PacketSize]);
//             
//             ei_pstore(&a[i+3*PacketSize], ei_padd(a3, b3));
//             a3 = ei_pload(&a[i+7*PacketSize]);
//             b3 = ei_pload(&b[i+7*PacketSize]);
//             
//             ei_pstore(&a[i+4*PacketSize], ei_padd(a0, b0));
//             ei_pstore(&a[i+5*PacketSize], ei_padd(a1, b1));
//             ei_pstore(&a[i+6*PacketSize], ei_padd(a2, b2));
//             ei_pstore(&a[i+7*PacketSize], ei_padd(a3, b3));
            
            ei_pstore(&a[i+2*PacketSize], ei_padd(ei_ploadu(&a[i+2*PacketSize]), ei_ploadu(&b[i+2*PacketSize])));
            ei_pstore(&a[i+3*PacketSize], ei_padd(ei_ploadu(&a[i+3*PacketSize]), ei_ploadu(&b[i+3*PacketSize])));
            ei_pstore(&a[i+4*PacketSize], ei_padd(ei_ploadu(&a[i+4*PacketSize]), ei_ploadu(&b[i+4*PacketSize])));
            ei_pstore(&a[i+5*PacketSize], ei_padd(ei_ploadu(&a[i+5*PacketSize]), ei_ploadu(&b[i+5*PacketSize])));
            ei_pstore(&a[i+6*PacketSize], ei_padd(ei_ploadu(&a[i+6*PacketSize]), ei_ploadu(&b[i+6*PacketSize])));
            ei_pstore(&a[i+7*PacketSize], ei_padd(ei_ploadu(&a[i+7*PacketSize]), ei_ploadu(&b[i+7*PacketSize])));
        }
}
