dnl $Id: ac_set_compiler.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl AC_SET_COMPILER: borrowed from blitz++
dnl

AC_DEFUN([AC_SET_COMPILER],
  [cxxwith=`echo $1 | sed -e 's/ /@/'`
   case "$cxxwith" in
     *:*@*)                 # Full initialization syntax
       CXX=`echo "$cxxwith" | sed  -n -e 's/.*:\(.*\)@.*/\1/p'`
       CXXFLAGS=`echo "$cxxwith" | sed  -n -e 's/.*:.*@\(.*\)/\1/p'`
     ;;
     *:*)                   # Simple initialization syntax
       CXX=`echo "$cxxwith" | sed  -n -e 's/.*:\(.*\)/\1/p'`
       CXXFLAGS=$3
     ;;
     *)                     # Default values
       CXX=$2
       CXXFLAGS=$3
     ;;
   esac])
