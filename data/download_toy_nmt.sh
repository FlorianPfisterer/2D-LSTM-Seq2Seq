#!/bin/sh

mkdir ./dataset
curl http://www.manythings.org/anki/deu-eng.zip | tar -xf - -C ./dataset

mv dataset/deu.txt ./eng-deu.txt
rm -rf dataset