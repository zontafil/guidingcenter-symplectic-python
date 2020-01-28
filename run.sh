#!/bin/sh
DATE=$(date +"%Y-%m-%d_%H:%M:%S")
SIM_PREFIX="sims"
GITTOKEN=$(git rev-parse HEAD | cut -c1-10)
FOLDERPREFIX=$SIM_PREFIX/"$DATE"_$GITTOKEN

mkdir -p $FOLDERPREFIX/out
mkdir -p $FOLDERPREFIX/src
mkdir -p $FOLDERPREFIX/plots
mkdir -p plots

echo "=== Starting simulation: saving to $FOLDERPREFIX"

echo "begin simulation "$DATE > $FOLDERPREFIX/info.txt
python3 src/main.py -o $FOLDERPREFIX/out/out.txt | tee $FOLDERPREFIX/info.txt
echo "end simulation "$(date +"%Y-%m-%d_%H:%M:%S") >> $FOLDERPREFIX/info.txt

echo "=== End simulation"
echo "=== Saving charts"

# save plots
python3 src/plot.py --oshort=$FOLDERPREFIX/plots/ --olong=plots/"$DATE"_

# if [ -f "Blines.txt" ]
# then
# 	mv Blines.txt $SIM_PREFIX/$1/
# fi

echo "=== Backing up sources"
cp -r src/* $FOLDERPREFIX/src
