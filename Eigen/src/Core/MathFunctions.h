// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

template<typename T> inline typename NumTraits<T>::Real precision();
template<typename T> inline T random(T a, T b);
template<typename T> inline T random();

template<> inline int precision<int>() { return 0; }
inline int real(int x)  { return x; }
inline int imag(int)    { return 0; }
inline int conj(int x)  { return x; }
inline int abs(int x)   { return std::abs(x); }
inline int abs2(int x)  { return x*x; }
inline int sqrt(int)
{
  // Taking the square root of integers is not allowed
  // (the square root does not always exist within the integers).
  // Please cast to a floating-point type.
  assert(false);
  return 0;
}
template<> inline int random(int a, int b)
{
  // We can't just do rand()%n as only the high-order bits are really random
  return a + static_cast<int>((b-a+1) * (rand() / (RAND_MAX + 1.0)));
}
template<> inline int random()
{
  return random<int>(-10, 10);
}
inline bool isMuchSmallerThan(int a, int, int = precision<int>())
{
  return a == 0;
}
inline bool isApprox(int a, int b, int = precision<int>())
{
  return a == b;
}
inline bool isApproxOrLessThan(int a, int b, int = precision<int>())
{
  return a <= b;
}

template<> inline float precision<float>() { return 1e-5f; }
inline float real(float x)  { return x; }
inline float imag(float)    { return 0.f; }
inline float conj(float x)  { return x; }
inline float abs(float x)   { return std::abs(x); }
inline float abs2(float x)  { return x*x; }
inline float sqrt(float x)  { return std::sqrt(x); }
template<> inline float random(float a, float b)
{
  return a + (b-a) * std::rand() / RAND_MAX;
}
template<> inline float random()
{
  return random<float>(-10.0f, 10.0f);
}
inline bool isMuchSmallerThan(float a, float b, float prec = precision<float>())
{
  return std::abs(a) <= std::abs(b) * prec;
}
inline bool isApprox(float a, float b, float prec = precision<float>())
{
  return std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * prec;
}
inline bool isApproxOrLessThan(float a, float b, float prec = precision<float>())
{
  return a <= b || isApprox(a, b, prec);
}

template<> inline double precision<double>() { return 1e-11; }
inline double real(double x)  { return x; }
inline double imag(double)    { return 0.; }
inline double conj(double x)  { return x; }
inline double abs(double x)   { return std::abs(x); }
inline double abs2(double x)  { return x*x; }
inline double sqrt(double x)  { return std::sqrt(x); }
template<> inline double random(double a, double b)
{
  return a + (b-a) * std::rand() / RAND_MAX;
}
template<> inline double random()
{
  return random<double>(-10.0, 10.0);
}
inline bool isMuchSmallerThan(double a, double b, double prec = precision<double>())
{
  return std::abs(a) <= std::abs(b) * prec;
}
inline bool isApprox(double a, double b, double prec = precision<double>())
{
  return std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * prec;
}
inline bool isApproxOrLessThan(double a, double b, double prec = precision<double>())
{
  return a <= b || isApprox(a, b, prec);
}

template<> inline float precision<std::complex<float> >() { return precision<float>(); }
inline float real(const std::complex<float>& x) { return std::real(x); }
inline float imag(const std::complex<float>& x) { return std::imag(x); }
inline std::complex<float> conj(const std::complex<float>& x) { return std::conj(x); }
inline float abs(const std::complex<float>& x) { return std::abs(x); }
inline float abs2(const std::complex<float>& x) { return std::norm(x); }
inline std::complex<float> sqrt(const std::complex<float>&)
{
  // Taking the square roots of complex numbers is not allowed,
  // as this is ambiguous (there are two square roots).
  // What were you trying to do?
  assert(false);
  return 0;
}
template<> inline std::complex<float> random()
{
  return std::complex<float>(random<float>(), random<float>());
}
inline bool isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b, float prec = precision<float>())
{
  return abs2(a) <= abs2(b) * prec * prec;
}
inline bool isMuchSmallerThan(const std::complex<float>& a, float b, float prec = precision<float>())
{
  return abs2(a) <= abs2(b) * prec * prec;
}
inline bool isApprox(const std::complex<float>& a, const std::complex<float>& b, float prec = precision<float>())
{
  return isApprox(std::real(a), std::real(b), prec)
      && isApprox(std::imag(a), std::imag(b), prec);
}
// isApproxOrLessThan wouldn't make sense for complex numbers

template<> inline double precision<std::complex<double> >() { return precision<double>(); }
inline double real(const std::complex<double>& x) { return std::real(x); }
inline double imag(const std::complex<double>& x) { return std::imag(x); }
inline std::complex<double> conj(const std::complex<double>& x) { return std::conj(x); }
inline double abs(const std::complex<double>& x) { return std::abs(x); }
inline double abs2(const std::complex<double>& x) { return std::norm(x); }
template<> inline std::complex<double> random()
{
  return std::complex<double>(random<double>(), random<double>());
}
inline bool isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b, double prec = precision<double>())
{
  return abs2(a) <= abs2(b) * prec * prec;
}
inline bool isMuchSmallerThan(const std::complex<double>& a, double b, double prec = precision<double>())
{
  return abs2(a) <= abs2(b) * prec * prec;
}
inline bool isApprox(const std::complex<double>& a, const std::complex<double>& b, double prec = precision<double>())
{
  return isApprox(std::real(a), std::real(b), prec)
      && isApprox(std::imag(a), std::imag(b), prec);
}
// isApproxOrLessThan wouldn't make sense for complex numbers

#define EIGEN_MAKE_MORE_OVERLOADED_COMPLEX_OPERATOR_STAR(T,U) \
inline std::complex<T> operator*(U a, const std::complex<T>& b) \
{ \
  return std::complex<T>(static_cast<T>(a)*b.real(), \
                         static_cast<T>(a)*b.imag()); \
} \
inline std::complex<T> operator*(const std::complex<T>& b, U a) \
{ \
  return std::complex<T>(static_cast<T>(a)*b.real(), \
                         static_cast<T>(a)*b.imag()); \
}

EIGEN_MAKE_MORE_OVERLOADED_COMPLEX_OPERATOR_STAR(int,    float)
EIGEN_MAKE_MORE_OVERLOADED_COMPLEX_OPERATOR_STAR(int,    double)
EIGEN_MAKE_MORE_OVERLOADED_COMPLEX_OPERATOR_STAR(float,  double)
EIGEN_MAKE_MORE_OVERLOADED_COMPLEX_OPERATOR_STAR(double, float)

#endif // EIGEN_MATHFUNCTIONS_H
