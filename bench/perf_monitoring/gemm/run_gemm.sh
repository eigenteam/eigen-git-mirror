#!/bin/bash

if [ ! -d "eigen_src" ]; then
  hg clone https://bitbucket.org/eigen/eigen eigen_src
fi

if [ ! -z '$CXX' ]; then
  CXX=g++
fi

rm sgemm.out
rm dgemm.out
rm cgemm.out

function test_current 
{
  rev=$1
  scalar=$2
  name=$3
  
  if $CXX -O2 -DNDEBUG -march=native $CXX_FLAGS -I eigen_src gemm.cpp -DSCALAR=$scalar -o $name; then
    res=`./$name`
    echo $res
    echo "$rev $res" >> $name.out
  else
    echo "Compilation failed, skip rev $rev"
  fi
}

while read rev
do
  if [ ! -z '$rev' ]; then
    echo "Testing rev $rev"
    cd eigen_src
    hg up -C $rev
    actual_rev=`hg identify | cut -f1 -d' '`
    cd ..
    
    test_current $actual_rev float                  sgemm
    test_current $actual_rev double                 dgemm
    test_current $actual_rev "std::complex<double>" cgemm
  fi
  
done < changesets.txt

echo "Float:"
cat sgemm.out
echo ""

echo "Double:"
cat dgemm.out
echo ""

echo "Complex:"
cat cgemm.out
echo ""

./make_plot.sh sgemm
./make_plot.sh dgemm
./make_plot.sh cgemm


