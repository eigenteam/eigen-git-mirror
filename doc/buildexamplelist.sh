#!/bin/sh

echo "namespace Eigen {"
echo "/** \page ExampleList"
echo "<h1>Selected list of examples</h1>"

grep \\addexample $1/Eigen/* -R | cut -d \\ -f 2- | \
while read example;
do
anchor=`echo "$example" | cut -d " " -f 2`
text=`echo "$example" | cut -d " " -f 4-`
echo "\\\li \\\ref $anchor \"$text\""
done
echo "*/"
echo "}"