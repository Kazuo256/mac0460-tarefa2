#!/bin/bash

outdir=tarefa2-wilson

if [ ! -d $outdir ]; then
  mkdir $outdir
fi

contents="tarefa2.py relatorio.txt LEIAME.txt sample test"
cp -r $contents $outdir

tar -caf $outdir.tar.gz $outdir

rm -rf $outdir

