#!/bin/bash
filename=${1:-github-java-files-train-TOP.txt}
filename_parsed='parsed-files.txt'
files_handled_log='files-handled.log'
json_files='json-files.txt'
temp_batch_file='temp-batch.txt'
stderr_file='error.log'

#Delete the Last Log
rm $files_handled_log $stderr_file output.json JSONFiles/*
# Appends 'java_projects' to beginning of each file_ptr
sed 's/^\./\.\/java_projects/g' $filename > $filename_parsed


java -jar ~/grammar_vae/java_compiler/tool_files/batch_dom_driver/target/batch_dom_driver-1.0-jar-with-dependencies.jar $filename_parsed config.json > $files_handled_log 2>$stderr_file
# Need to stich the jsons after this step
#locate all json files easily by this
#sed -i 's/^.*\//JSONFiles\//g' $filename_parsed
sed 's/$/\.json/g' $filename_parsed > $json_files
# Use the prebuilt merge API
python3 merge.py $json_files --output_file output_temp.json
#export PYTHONPATH=~/bayou/src/main/python
#python3 bayou/src/main/python/scripts/evidence_extractor.py output_temp.json output.json
#delete temp files
#rm $temp_batch_file $json_files $filename_parsed
