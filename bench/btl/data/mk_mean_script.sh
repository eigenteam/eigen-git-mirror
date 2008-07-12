#! /bin/bash
WHAT=$1
DIR=$2
MINIC=$3
MAXIC=$4
MINOC=$5
MAXOC=$6

WORK_DIR=tmp
mkdir $WORK_DIR

DATA_FILE=`find $DIR -name "*.dat" | grep _${WHAT}`
echo
for FILE in $DATA_FILE
do
        ##echo hello world
        ##echo "mk_mean_script1" ${FILE}
	BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}

	##echo "mk_mean_script1" ${TITLE}
	cp $FILE ${WORK_DIR}/${TITLE}

done

cd $WORK_DIR
../main $1 $3 $4 $5 $6 *
../mk_new_gnuplot.sh $1 $2
rm -f *.gnuplot
cd ..

rm -R $WORK_DIR

webpagefilename=$2/index.html
# echo '<h3>'${WHAT}'</h3>'  >> $webpagefilename
echo '<a href="/btl/'$1'.pdf"><img src="/btl/'$1'.png" alt="'${WHAT}'" /></a><br/>'  >> $webpagefilename








