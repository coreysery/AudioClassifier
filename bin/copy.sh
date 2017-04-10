#!/usr/bin/env bash

# Send project to belmont server

rsync $HOME/Developer/ai/DeepDSP/ csery@hedges.belmont.edu:/home/csery/DeepDSP \
--verbose \
--recursive \
--update \
--compress \
--exclude=*resources/audio/* \
--exclude=*resources/tmp/* \
--exclude=*resources/source/* \
--exclude=*.git/* \
--exclude=*deep/* \
--progress \
