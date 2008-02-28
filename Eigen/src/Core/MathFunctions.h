// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

template<typename T> inline typename NumTraits<T>::Real precision();
template<typename T> inline T ei_random(T a, T b);
template<typename T> inline T ei_random();

template<> inline int precision<int>() { return 0; }
inline int ei_real(int x)  { return x; }
inline int ei_imag(int)    { return 0; }
inline int ei_conj(int x)  { return x; }
inline int ei_abs(int x)   { return abs(x); }
inline int ei_abs2(int x)  { return x*x; }
inline int ei_sqrt(int)
{
  // Taking the square root of integers is not allowed
  // (the square root does not always exist within the integers).
  // Please cast to a floating-point type.
  assert(false);
  return 0;
}
template<> inline int ei_random(int a, int b)
{
  // We can't just do rand()%n as only the high-order bits are really random
  return a + static_cast<int>((b-a+1) * (rand() / (RAND_MAX + 1.0)));
}
template<> inline int ei_random()
{
  return ei_random<int>(-10, 10);
}
inline bool ei_isMuchSmallerThan(int a, int, int = precision<int>())
{
  return a == 0;
}
inline bool ei_isApprox(int a, int b, int = precision<int>())
{
  return a == b;
}
inline bool ei_isApproxOrLessThan(int a, int b, int = precision<int>())
{
  return a <= b;
}

template<> inline float precision<float>() { return 1e-5f; }
inline float ei_real(float x)  { return x; }
inline float ei_imag(float)    { return 0.f; }
inline float ei_conj(float x)  { return x; }
inline float ei_abs(float x)   { return std::abs(x); }
inline float ei_abs2(float x)  { return x*x; }
inline float ei_sqrt(float x)  { return std::sqrt(x); }
template<> inline float ei_random(float a, float b)
{
  return a + (b-a) * std::rand() / RAND_MAX;
}
template<> inline float ei_random()
{
  return ei_random<float>(-10.0f, 10.0f);
}
inline bool ei_isMuchSmallerThan(float a, float b, float prec = precision<float>())
{
  return ei_abs(a) <= ei_abs(b) * prec;
}
inline bool ei_isApprox(float a, float b, float prec = precision<float>())
{
  return ei_abs(a - b) <= std::min(ei_abs(a), ei_abs(b)) * prec;
}
inline bool ei_isApproxOrLessThan(float a, float b, float prec = precision<float>())
{
  return a <= b || ei_isApprox(a, b, prec);
}

template<> inline double precision<double>() { return 1e-11; }
inline double ei_real(double x)  { return x; }
inline double ei_imag(double)    { return 0.; }
inline double ei_conj(double x)  { return x; }
inline double ei_abs(double x)   { return std::abs(x); }
inline double ei_abs2(double x)  { return x*x; }
inline double ei_sqrt(double x)  { return std::sqrt(x); }
template<> inline double ei_random(double a, double b)
{
  return a + (b-a) * std::rand() / RAND_MAX;
}
template<> inline double ei_random()
{
  return ei_random<double>(-10.0, 10.0);
}
inline bool ei_isMuchSmallerThan(double a, double b, double prec = precision<double>())
{
  return ei_abs(a) <= ei_abs(b) * prec;
}
inline bool ei_isApprox(double a, double b, double prec = precision<double>())
{
  return ei_abs(a - b) <= std::min(ei_abs(a), ei_abs(b)) * prec;
}
inline bool ei_isApproxOrLessThan(double a, double b, double prec = precision<double>())
{
  return a <= b || ei_isApprox(a, b, prec);
}

template<> inline float precision<std::complex<float> >() { return precision<float>(); }
inline float ei_real(const std::complex<float>& x) { return std::real(x); }
inline float ei_imag(const std::complex<float>& x) { return std::imag(x); }
inline std::complex<float> ei_conj(const std::complex<float>& x) { return std::conj(x); }
inline float ei_abs(const std::complex<float>& x) { return std::abs(x); }
inline float ei_abs2(const std::complex<float>& x) { return std::norm(x); }
inline std::complex<float> ei_sqrt(const std::complex<float>&)
{
  // Taking the square roots of complex numbers is not allowed,
  // as this is ambiguous (there are two square roots).
  // What were you trying to do?
  assert(false);
  return 0;
}
template<> inline std::complex<float> ei_random()
{
  return std::complex<float>(ei_random<float>(), ei_random<float>());
}
inline bool ei_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b, float prec = precision<float>())
{
  return ei_abs2(a) <= ei_abs2(b) * prec * prec;
}
inline bool ei_isMuchSmallerThan(const std::complex<float>& a, float b, float prec = precision<float>())
{
  return ei_abs2(a) <= ei_abs2(b) * prec * prec;
}
inline bool ei_isApprox(const std::complex<float>& a, const std::complex<float>& b, float prec = precision<float>())
{
  return ei_isApprox(ei_real(a), ei_real(b), prec)
      && ei_isApprox(ei_imag(a), ei_imag(b), prec);
}
// ei_isApproxOrLessThan wouldn't make sense for complex numbers

template<> inline double precision<std::complex<double> >() { return precision<double>(); }
inline double ei_real(const std::complex<double>& x) { return std::real(x); }
inline double ei_imag(const std::complex<double>& x) { return std::imag(x); }
inline std::complex<double> ei_conj(const std::complex<double>& x) { return std::conj(x); }
inline double ei_abs(const std::complex<double>& x) { return std::abs(x); }
inline double ei_abs2(const std::complex<double>& x) { return std::norm(x); }
template<> inline std::complex<double> ei_random()
{
  return std::complex<double>(ei_random<double>(), ei_random<double>());
}
inline bool ei_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b, double prec = precision<double>())
{
  return ei_abs2(a) <= ei_abs2(b) * prec * prec;
}
inline bool ei_isMuchSmallerThan(const std::complex<double>& a, double b, double prec = precision<double>())
{
  return ei_abs2(a) <= ei_abs2(b) * prec * prec;
}
inline bool ei_isApprox(const std::complex<double>& a, const std::complex<double>& b, double prec = precision<double>())
{
  return ei_isApprox(ei_real(a), ei_real(b), prec)
      && ei_isApprox(ei_imag(a), ei_imag(b), prec);
}
// ei_isApproxOrLessThan wouldn't make sense for complex numbers

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
