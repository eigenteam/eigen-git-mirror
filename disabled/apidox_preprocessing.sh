#!/bin/bash

CXX=`which g++`
SRC=$1
mkdir -p eigen2/out

if expr match $SRC ".*\/examples\/.*" > /dev/null ; then

#   DST=`echo $SRC | sed 's/examples/out/' | sed 's/cpp$/out/'`
  DST=`echo $SRC | sed 's/.*\/examples/eigen2\/out/' | sed 's/cpp$/out/'`
  INC=`echo $SRC | sed 's/\/doc\/examples\/.*/\//'`

  if ! test -e $DST || test $SRC -nt $DST ; then
    $CXX $SRC -I. -I$INC -o eitmp_example && ./eitmp_example > $DST
    rm eitmp_example
  fi

elif expr match $SRC ".*\/snippets\/.*" > /dev/null ; then

#   DST=`echo $SRC | sed 's/snippets/out/' | sed 's/cpp$/out/'`
  DST=`echo $SRC | sed 's/.*\/snippets/eigen2\/out/' | sed 's/cpp$/out/'`
  INC=`echo $SRC | sed 's/\/doc\/snippets\/.*/\//'`

  if ! test -e $DST || test $SRC -nt $DST ; then
    echo "#include <Eigen/Core>" > .ei_in.cpp
    echo "#include <Eigen/Array>" >> .ei_in.cpp
    echo "#include <Eigen/LU>" >> .ei_in.cpp
    echo "#include <Eigen/Cholesky>" >> .ei_in.cpp
    echo "#include <Eigen/Geometry>" >> .ei_in.cpp
    echo "using namespace Eigen; using namespace std;" >> .ei_in.cpp
    echo "int main(int, char**){cout.precision(3);" >> .ei_in.cpp
    cat $SRC >> .ei_in.cpp
    echo "return 0;}" >> .ei_in.cpp
    echo " " >> .ei_in.cpp
    
    $CXX .ei_in.cpp -I. -I$INC -o eitmp_example && ./eitmp_example > $DST
    rm eitmp_example
    rm .ei_in.cpp
  fi

fi

cat $SRC
exit 0
