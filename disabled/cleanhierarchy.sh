#!/bin/sh

sed -i 's/^.li.*MatrixBase\&lt.*gt.*a.$/ /g' $1
sed -i 's/^.li.*MapBase\&lt.*gt.*a.$/ /g' $1
sed -i 's/^.li.*RotationBase\&lt.*gt.*a.$/ /g' $1
