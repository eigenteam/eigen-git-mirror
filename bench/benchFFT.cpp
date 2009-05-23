// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
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

#include <complex>
#include <vector>
#include <Eigen/Core>
#include <bench/BenchTimer.h>
#include <unsupported/Eigen/FFT.h>

using namespace Eigen;
using namespace std;

#ifndef NFFT
#define NFFT 1024
#endif

#ifndef TYPE
#define TYPE float
#endif

#ifndef NITS 
#define NITS (10000000/NFFT)
#endif

int main() 
{
  vector<complex<TYPE> > inbuf(NFFT);
  vector<complex<TYPE> > outbuf(NFFT);
  Eigen::FFT<TYPE> fft;

  fft.fwd( outbuf , inbuf);

  BenchTimer timer;
  timer.reset();
  for (int k=0;k<8;++k) {
      timer.start();
      for(int i = 0; i < NITS; i++)
          fft.fwd( outbuf , inbuf);
      timer.stop();
  }
  double mflops = 5.*NFFT*log2((double)NFFT) / (1e6 * timer.value() / (double)NITS );
  cout << "NFFT=" << NFFT << "  " << (double(1e-6*NFFT*NITS)/timer.value()) << " MS/s  " << mflops << "MFLOPS\n";
}
