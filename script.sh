#!/bin/bash

cd notebooks

for dir in */
do
  rm -r ../docs/nb_img/$dir
  rm -r ../docs/$dir
  for nb in $dir*.ipynb
  do
    jupyter nbconvert --to markdown $nb --output-dir ../docs/$dir --NbConvertApp.output_files_dir="../nb_img/$dir"

    jupyter nbconvert --to script $nb --output-dir ../scripts/$dir

  done
done

cd ..

echo maybe you want to run
echo mkdocs gh-deploy
echo to update the website