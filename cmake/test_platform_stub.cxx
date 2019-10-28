#include "cmath"

int main()
{
  int i=1;
  float a=std::tan(3.0);
  double b=std::cos(4.0);
  return int( std::fmin( 0.0, std::fmax( int( std::sin( ( i  * a + b )/ 12.0 )  ) * 10, 120) ) );
}
