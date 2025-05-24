#!/bin/bash

# Script to remove the datestamp range from omni2, sat_density, and goes filenames

# Assumes you are cd into a directory that has "omni2", "sat_density", and "goes" subdirectories, with
# each containing the csvs that were just unzipped from their respective Dropbox zipfiles.
#
# i.e. you have already set the files up like this
#
# mkdir -p /somefolder/data/omni2
# cd /somefolder/data/omni2
# unzip /omni2.zip -d .
# cd /somefolder/data
# Now you can run truncate_date.sh
#
# Unfortunately it's kind of a slow process.
#

# Nullglob means don't pass the glob as a literal value in the case where no files are matched by glob
shopt -s nullglob

pushd omni2 || exit
# Renames omni2 csv files to remove the date portion
for file in omni2-*.csv; do
  newname=$(echo "${file}" | sed -e 's/\(omni2-[0-9]*\)-.*\.csv/\1.csv/g')
  mv "${file}" "${newname}"
done
popd || exit

pushd sat_density || exit
# Renames sat_density csv files to remove the date portion
for file in *.csv; do
  newname=$(echo "${file}" | sed -E -e 's/(.{6}-[0-9]*)-.*\.csv/\1.csv/g')
  mv "${file}" "${newname}"
done
popd || exit

pushd goes || exit
# Renames goes csv files to remove the date portion
for file in *.csv; do
  newname=$(echo "${file}" | sed -E -e 's/(goes-[0-9]*)-.*\.csv/\1.csv/g')
  mv "${file}" "${newname}"
done
popd || exit