#!/bin/sh

mkdir -p saved_plots

LASTPLOT=$(ls -t plots | sed -n '1!p' | grep -v last | head -1)
cp plots/"$LASTPLOT" saved_plots/