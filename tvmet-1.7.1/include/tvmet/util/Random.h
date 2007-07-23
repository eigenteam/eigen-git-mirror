/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: SelfTest.cc,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_UTIL_RANDOM_H
#define TVMET_UTIL_RANDOM_H

#ifdef __GNUC__
# if __GNUC__>=4
#  define EIGEN_WITH_GCC_4_OR_LATER
# endif
#endif

#include <cstdlib>

#ifdef EIGEN_USE_COMPLEX
#include <complex>
#endif

namespace tvmet {

namespace util {

/** Stores in x a random int between -RAND_MAX/2 and RAND_MAX/2 */
inline void pickRandom( int & x )
{
    x = rand() - RAND_MAX / 2;
}


/** Stores in x a random float between -1.0 and 1.0 */
inline void pickRandom( float & x )
{
    x = 2.0f * rand() / RAND_MAX - 1.0f;
}

/** Stores in x a random double between -1.0 and 1.0 */
inline void pickRandom( double & x )
{
    x = 2.0 * rand() / RAND_MAX - 1.0;
}

#ifdef EIGEN_USE_COMPLEX
/** Stores in the real and imaginary parts of x
  * random values between -1.0 and 1.0 */
template<typename T> void pickRandom( std::complex<T> & x )
{
#ifdef EIGEN_WITH_GCC_4_OR_LATER
    pickRandom( x.real() );
    pickRandom( x.imag() );
#else // workaround by David Faure for MacOS 10.3 and GCC 3.3, commit 630812
    T r = x.real();
    T i = x.imag();
    pickRandom( r );
    pickRandom( i );
    x = std::complex<T>(r,i);
#endif
}
#endif // EIGEN_USE_COMPLEX

template<typename T> T someRandom()
{
  T t;
  pickRandom(t);
  return t;
}

} // namespace util

} // namespace tvmet

#endif // TVMET_UTIL_RANDOM_H
