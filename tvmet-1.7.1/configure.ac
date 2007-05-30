dnl $Id: configure.ac,v 1.47 2005/04/26 15:07:00 opetzold Exp $
dnl Configure script for the Tiny Vector Matrix Expression Template Library
dnl Process this file with autoconf to produce a configure script.


dnl autoconf 2.50 or higher to rebuild aclocal.m4, because the
dnl AC_CREATE_PREFIX_CONFIG_H macro needs the AS_DIRNAME macro.
AC_PREREQ(2.59)


dnl Copy revision stamp into the `configure' script,
AC_REVISION($Revision: 1.47 $)


dnl Disable caching.
define([AC_CACHE_LOAD], )
define([AC_CACHE_SAVE], )


dnl Process this file with autoconf to produce a configure script.
AC_INIT([tvmet],[1.7.1],[opetzold@users.sourceforge.net])


dnl inform about
AC_MSG_RESULT([Configure Tiny Vector Matrix using Expression Templates library AC_PACKAGE_VERSION])


dnl config and versioning, the config.h header is in config - we use a prefixed
dnl config header, see below
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_HEADERS([config/config.h])


dnl Getting the Canonical System Type
AC_CANONICAL_HOST


dnl require gnu style, automake 1.8 or better, etc.
AM_INIT_AUTOMAKE([gnu 1.8 no-define dist-bzip2])


dnl --enable-maintainer-mode for configure
AM_MAINTAINER_MODE


# Declare variables which we want substituted in the Makefile.in's
AC_SUBST(CXX_OPTIMIZE_FLAGS)
AC_SUBST(CXX_DEBUG_FLAGS)
AC_SUBST(CXX_WARN_FLAGS)


dnl Manual configuration options of oopattern.
AC_ARG_ENABLE(tvmet, [
Options of tvmet:])


dnl Process command-line arguments for compiler (parts borrowed from blitz++)
AC_MSG_CHECKING([for compiler])
AC_ARG_WITH(cxx,
  [  --with-cxx=compiler     set specific C++ compiler and options (e.g. g++ (default), KCC, icc, pgCC)],
  [case "$withval" in
     KCC*)               # KAI C++  http://www.kai.com/
       AC_MSG_RESULT([KAI C++])
       CXX_OPTIMIZE_FLAGS="-DTVMET_OPTIMIZE +K3 -O3 -fast --inline_keyword_space_time=10000"
       CXX_DEBUG_FLAGS="-g +K0 -DTVMET_DEBUG"
       CXX_WARN_FLAGS=""
     ;;
     g++*|gcc*)          # gcc  http://gcc.gnu.org/
       AC_MSG_RESULT([GNU C++])
       AC_SET_COMPILER($withval,"g++","")
       CXX_OPTIMIZE_FLAGS="-DTVMET_OPTIMIZE -O2 -finline-limit=5000 -ftemplate-depth-200 -fomit-frame-pointer"
       CXX_DEBUG_FLAGS="-g -O1 -DTVMET_DEBUG"
       CXX_WARN_FLAGS="-W -Wall -Winline"

       # paranoia on older gcc versions
       gxx_version=`$CXX -v 2>&1 | grep "^.*g.. version" | \
       sed -e 's/^.*g.. version *//' -e 's/cygnus-//' -e 's/egcs-//'`

       case "$gxx_version" in
           1.* | 2.[[0123456789]].*)
           AC_MSG_ERROR([g++ version $gxx_version will probably fail to compile tvmet])
         ;;
       esac
     ;;
     icc)                # Intel C++ compiler   http://www.intel.com/
       AC_MSG_RESULT([Intel C++])
       AC_SET_COMPILER($withval,"icc","")
       CXX_OPTIMIZE_FLAGS="-DTVMET_OPTIMIZE -O2 -fp_port -Qoption,c,-ip_ninl_max_total_stats=3000"
       CXX_DEBUG_FLAGS="-g -O0 -DTVMET_DEBUG"
       CXX_WARN_FLAGS="-Wall"
     ;;
     pgCC*)              # Portland group   http://www.pgroup.com/
       AC_MSG_RESULT([Portland Group C++])
       AC_SET_COMPILER($withval,"pgCC","")
       CXX_OPTIMIZE_FLAGS="-DTVMET_OPTIMIZE -O4 -Mnoframe -Mnodepchk -Minline=levels:25"
       CXX_DEBUG_FLAGS="-g -O0 -DTVMET_DEBUG"
       CXX_WARN_FLAGS=""
     ;;
     *)
        AC_MSG_WARN([--with-cxx: unknown c++ compiler $withval])
     ;;
   esac],[
       AC_MSG_RESULT([GNU C++ (default)])
       AC_SET_COMPILER($withval,"g++","")
       CXX_OPTIMIZE_FLAGS="-DTVMET_OPTIMIZE -O2 -finline-limit=5000 -ftemplate-depth-200 -fomit-frame-pointer"
       CXX_DEBUG_FLAGS="-g -O1 -DTVMET_DEBUG"
       CXX_WARN_FLAGS="-W -Wall -Winline"
   ])


dnl Check if the script was called with --enable-optimize, and if so
dnl modify the compilation flags.
AC_MSG_CHECKING([for compiler optimizations])
tvmet_optimize=0
AC_ARG_ENABLE(optimize,
  [  --enable-optimize       enable compiler optimization flags],
  [if test "$enableval" = yes; then
     AC_MSG_RESULT([yes])
     CXXFLAGS="$CXXFLAGS $CXX_OPTIMIZE_FLAGS"
     tvmet_optimize=1
   else
     AC_MSG_RESULT([no])
   fi], [
     AC_MSG_RESULT([no])
   ])


dnl Set platform-specific optimization settings
if test $tvmet_optimize -eq 1; then
  AC_MSG_CHECKING([for compiler platform-specific optimizations])
  case "$build_os" in
    *linux*)
      case "$CXX" in
        g++)
	  AC_MSG_RESULT([GNU/Linux (CPU=$build_cpu) using GNU GCC])
	  # specifying -march=cpu-type implies -mcpu=cpu-type.
          case "$build_cpu" in
            i386|i486)
	  	CXXFLAGS="$CXXFLAGS -malign-double -march=i386"
	    ;;
	    i586)
		CXXFLAGS="$CXXFLAGS -malign-double -march=i586 -msse"
	    ;;
	    i686)
		CXXFLAGS="$CXXFLAGS -malign-double -march=i686 -msse2"
	    ;;
	    k6*)
                CXXFLAGS="$CXXFLAGS -malign-double -march=k6 -m3dnow"
	  esac
        ;;
        icc)
  	  AC_MSG_RESULT([GNU/Linux (CPU=$build_cpu) using Intel CC])
          case "$build_cpu" in
	    i586)
		# Intel Pentium Pro, Intel Pentium II, and Intel Pentium III processors
		CXXFLAGS="$CXXFLAGS -tpp6"
	    ;;
	    i686)
		# Intel Pentium 4 processors
		CXXFLAGS="$CXXFLAGS -tpp7"
	    ;;
	  esac
        ;;
      esac
    ;;
    *)
      AC_MSG_RESULT([tvmet has not been tuned for this platform])
    ;;
  esac
fi


dnl Check if the script was called with --enable-debug, and if so
dnl modify the compilation flags.
AC_MSG_CHECKING([for compiler debug flags])
AC_ARG_ENABLE(debug,
  [  --enable-debug          enable compiler debug flags],
  [if test "$enableval" = yes; then
     AC_MSG_RESULT([yes])
     CXXFLAGS="$CXXFLAGS $CXX_DEBUG_FLAGS"
   fi], [
     AC_MSG_RESULT([no])
   ])


dnl Check if the script was called with --enable-warnings, and if so
dnl modify the compilation flags.
AC_MSG_CHECKING([for compiler warn flags])
AC_ARG_ENABLE(warnings,
  [  --enable-warnings       enable compiler warning flags],
  [if test "$enableval" = yes; then
     AC_MSG_RESULT([yes])
     CXXFLAGS="$CXXFLAGS $CXX_WARN_FLAGS"
   fi], [
     AC_MSG_RESULT([no])
  ])


dnl doxygen docs
OP_DOXYGEN_DOC


dnl cppunit support
oopattern_config_cppunit=yes
AC_ARG_ENABLE(cppunit,
  [  --disable-cppunit       disable unit tests using cppunit],
  oopattern_config_cppunit=$enableval)


dnl check for cppunit for unit tests
if test x"$oopattern_config_cppunit" = xyes; then

  dnl Check if we have libcppunit
  AM_PATH_CPPUNIT(1.8.0)

  if test "x$no_cppunit" = xyes; then
    CONFIG_CPPUNIT=no
  else
    CONFIG_CPPUNIT=yes
  fi

fi
AC_MSG_CHECKING(for libcppunit support)
if test "x${oopattern_config_cppunit}" = xyes -a "x$CONFIG_CPPUNIT" = xyes; then
  CONFIG_CPPUNIT=yes
  AC_MSG_RESULT([yes])
else
  CONFIG_CPPUNIT=no
  AC_MSG_RESULT([no])
fi
AM_CONDITIONAL(CONFIG_CPPUNIT,[test x"$CONFIG_CPPUNIT" = xyes])


dnl Checks for programs.
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_LANG([C++])
AC_PROG_CXX
AC_PROG_LN_S


dnl don't be confused, we don't build shared/static libs; it's
dnl for the head install process only (for build and install
dnl process from separate build directory)
AC_PROG_LIBTOOL


dnl Checks for libraries.
AC_CHECK_LIB(dl, main)


dnl Checks for header files.
AC_HEADER_STDC
AC_HEADER_STDBOOL			dnl for unit tests
AC_CHECK_HEADERS([sys/time.h unistd.h])


# Checks for library functions.
AC_CHECK_FUNCS([floor pow rint sqrt])


dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_STRUCT_TM

AC_C_LONG_DOUBLE
AC_C_LONG_LONG
AC_C_RESTRICT
AC_CXX_HAVE_MUTABLE
AC_CXX_TYPENAME
AC_CXX_PARTIAL_SPECIALIZATION
AC_CXX_NAMESPACES
AC_CXX_HAVE_IEEE_MATH
AC_CXX_HAVE_SYSV_MATH
AC_CXX_HAVE_COMPLEX
AC_CXX_HAVE_COMPLEX_MATH1
AC_CXX_HAVE_COMPLEX_MATH2


dnl create makefiles
AC_CONFIG_FILES([
	Makefile
	tvmet-config
	tvmet.spec
	include/Makefile
	include/tvmet/Makefile
	include/tvmet/config/Makefile
	include/tvmet/meta/Makefile
	include/tvmet/loop/Makefile
	include/tvmet/util/Makefile
	include/tvmet/xpr/Makefile
	include/tvmet/config/config-vc71.h
	doc/Makefile
	doc/Doxyfile
	doc/footer.html
	doc/tvmet.sty
	testsuite/Makefile
	tests/Makefile
	examples/Makefile
	benchmark/Makefile
])
AC_CONFIG_COMMANDS([default],[[chmod a+x tvmet-config]],[[]])
AC_OUTPUT


dnl prefixed config header
AC_CREATE_PREFIX_CONFIG_H([include/tvmet/config.h], $PACKAGE, [config/config.h])


dnl -------------------
dnl tvmet's user epilog
dnl -------------------
echo "
Configuration for  $PACKAGE $VERSION:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following optional features are enabled:

Source code location          : $srcdir
tvmet's unit-tests available  : `if test "x$CONFIG_CPPUNIT" = xyes; then echo yes; else echo no; fi`
build documentation           : `if test "x$CONFIG_DOC" = xyes; then echo yes; else echo no; fi`
using LaTeX non-stop mode     : `if test "x$CONFIG_DOC_LATEX_NONSTOP" = xyes; then echo yes; else echo no; fi`
graphs and diagrams           : `if test "x$DOXYGEN_HAVE_DOT" = xyes; then echo -n yes; else echo no; fi`
Install path                  : $prefix

Compiler Flags used:

CXX                           : $CXX
CXXFLAGS                      : $CXXFLAGS

The configure script has generated makefiles for your platform.
You have several options now:

make               Depends on configure's flags given, build the docs,
                   build $PACKAGE $VERSION and run the regression tests.
make docs          Makes the docs using doxygen (if enabled).
make check         Make the unit tests (if CppUnit available).
make install       Install tvmet.
make uninstall     Uninstall tvmet.
"
