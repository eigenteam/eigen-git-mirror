#! /bin/bash
WHAT=$1
DIR=$2

cat ../gnuplot_common_settings.hh > ${WHAT}.gnuplot
cat ../${WHAT}.hh >> ${WHAT}.gnuplot

DATA_FILE=`cat ../order_lib`
echo set term postscript color rounded enhanced >> $WHAT.gnuplot
echo set output "'"../${DIR}/$WHAT.ps"'" >> $WHAT.gnuplot

echo plot \\ >> $WHAT.gnuplot

for FILE in $DATA_FILE
do
    LAST=$FILE
done

for FILE in $DATA_FILE
do
    BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}

    echo "'"$FILE"'" `grep $TITLE ../perlib_plot_settings.txt | head -n 1 | cut -d ";" -f 2` "\\" >>  $WHAT.gnuplot
    if [ $FILE != $LAST ]
    then
      echo ", \\" >>  $WHAT.gnuplot
    fi
done
echo " " >>  $WHAT.gnuplot

gnuplot -persist < $WHAT.gnuplot

rm $WHAT.gnuplot

ps2pdf ../${DIR}/$WHAT.ps ../${DIR}/$WHAT.pdf
convert -density 120 -rotate 90 -resize 800 +dither -colors 48 -quality 0 ../${DIR}/$WHAT.ps ../${DIR}/$WHAT.png
