#!/usr/bin/env bash

echo "Running Downsample"

cd "$(dirname "$0")"
mkdir ../resources/tmp

for dir in ../resources/audio/** ; do
  type=$(basename "$dir" .deb)
  dest="../resources/tmp/${type}"

  mkdir $dest

  for file in $dir/*.wav ; do
    b=$(basename "$file" .deb)

    if [ ! -e "${dest}/${b}" ] # If file does not exist yet
        then
            sox "${file}" -b 16 -c 1 -r 11025 --no-dither --show-progress "${dest}/${b}"
    fi;

  done;

done;

