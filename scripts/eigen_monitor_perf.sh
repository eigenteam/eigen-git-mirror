#!/bin/bash

# This is a script example to automatically update and upload performance unit tests.
# The following five variables must be adjusted to match your settings.

USER='ggael'
UPLOAD_DIR=perf_monitoring/ggaelmacbook26
EIGEN_SOURCE_PATH=$HOME/Eigen/eigen
export PREFIX="haswell-fma-"
export CXX_FLAGS="-mfma"

$EIGEN_SOURCE_PATH/bench/perf_monitoring/gemm/runall.sh $*

# (the '/' at the end of path is very important, see rsync documentation)
rsync -az --no-p --delete $EIGEN_SOURCE_PATH/bench/perf_monitoring/gemm/haswell-fma-*.png $USER@ssh.tuxfamily.org:eigen/eigen.tuxfamily.org-web/htdocs/$UPLOAD_DIR/ || { echo "upload failed"; exit 1; }

# fix the perm
ssh $USER@ssh.tuxfamily.org "chmod -R g+w /home/eigen/eigen.tuxfamily.org-web/htdocs/perf_monitoring" || { echo "perm failed"; exit 1; }
