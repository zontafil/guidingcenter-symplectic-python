#/bin/sh

OLDERTHAN=$1

if [ -z ${OLDERTHAN} ]; then
    OLDERTHAN=7
fi

echo "Deleting data older than $OLDERTHAN days"

find sims/ -name "out.txt" -mtime $OLDERTHAN -delete

