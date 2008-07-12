#! /bin/bash
WHAT=$1
DIR=$2

cat ../gnuplot_common_settings.hh > ${WHAT}.gnuplot
cat ../${WHAT}.hh >> ${WHAT}.gnuplot

DATA_FILE=`cat ../order_lib`

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


echo set term postscript color >> $WHAT.gnuplot
echo set output "'"../${DIR}/$WHAT.ps"'" >> $WHAT.gnuplot
echo replot >> $WHAT.gnuplot

echo set term png truecolor size 800,600 >> $WHAT.gnuplot
echo set output "'"../${DIR}/$WHAT.png"'" >> $WHAT.gnuplot
echo replot >> $WHAT.gnuplot


gnuplot -persist < $WHAT.gnuplot

rm $WHAT.gnuplot

# echo "`pwd` hh s2pdf $WHAT.ps $WHAT.pdf" > ../log.txt

ps2pdf ../${DIR}/$WHAT.ps ../${DIR}/$WHAT.pdf




