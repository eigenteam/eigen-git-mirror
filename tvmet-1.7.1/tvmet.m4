dnl $Id: tvmet.m4,v 1.3 2004/04/23 21:03:29 opetzold Exp $

dnl
dnl AM_PATH_TVMET([MINIMUM-VERSION, [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]]])
dnl
AC_DEFUN([AM_PATH_TVMET],
[

AC_ARG_WITH(tvmet-prefix,[  --with-tvmet-prefix=PFX   Prefix where tvmet is installed (optional)],
            tvmet_config_prefix="$withval", tvmet_config_prefix="")
AC_ARG_WITH(tvmet-exec-prefix,[  --with-tvmet-exec-prefix=PFX  Exec prefix where tvmet is installed (optional)],
            tvmet_config_exec_prefix="$withval", tvmet_config_exec_prefix="")

  if test x$tvmet_config_exec_prefix != x ; then
     tvmet_config_args="$tvmet_config_args --exec-prefix=$tvmet_config_exec_prefix"
     if test x${TVMET_CONFIG+set} != xset ; then
        TVMET_CONFIG=$tvmet_config_exec_prefix/bin/tvmet-config
     fi
  fi
  if test x$tvmet_config_prefix != x ; then
     tvmet_config_args="$tvmet_config_args --prefix=$tvmet_config_prefix"
     if test x${TVMET_CONFIG+set} != xset ; then
        TVMET_CONFIG=$tvmet_config_prefix/bin/tvmet-config
     fi
  fi

  AC_PATH_PROG(TVMET_CONFIG, tvmet-config, no)
  tvmet_version_min=$1

  AC_MSG_CHECKING(for tvmet - version >= $tvmet_version_min)
  no_tvmet=""
  if test "$TVMET_CONFIG" = "no" ; then
    no_tvmet=yes
  else
    TVMET_CXXFLAGS=`$TVMET_CONFIG --cxxflags`
    TVMET_LIBS=`$TVMET_CONFIG --libs`
    tvmet_version=`$TVMET_CONFIG --version`

    tvmet_major_version=`echo $tvmet_version | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
    tvmet_minor_version=`echo $tvmet_version | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
    tvmet_micro_version=`echo $tvmet_version | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`

    tvmet_major_min=`echo $tvmet_version_min | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
    tvmet_minor_min=`echo $tvmet_version_min | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
    tvmet_micro_min=`echo $tvmet_version_min | \
           sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`

    tvmet_version_proper=`expr \
        $tvmet_major_version \> $tvmet_major_min \| \
        $tvmet_major_version \= $tvmet_major_min \& \
        $tvmet_minor_version \> $tvmet_minor_min \| \
        $tvmet_major_version \= $tvmet_major_min \& \
        $tvmet_minor_version \= $tvmet_minor_min \& \
        $tvmet_micro_version \>= $tvmet_micro_min `

    if test "$tvmet_version_proper" = "1" ; then
      AC_MSG_RESULT([$tvmet_major_version.$tvmet_minor_version.$tvmet_micro_version])
    else
      AC_MSG_RESULT(no)
      no_tvmet=yes
    fi
  fi

  if test "x$no_tvmet" = x ; then
     ifelse([$2], , :, [$2])
  else
     TVMET_CXXFLAGS=""
     TVMET_LIBS=""
     ifelse([$3], , :, [$3])
  fi

  AC_SUBST(TVMET_CXXFLAGS)
  AC_SUBST(TVMET_LIBS)
])
