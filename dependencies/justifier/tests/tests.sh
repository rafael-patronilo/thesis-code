#!/usr/bin/env bash

run_test() {
  echo "Running run_test $1"
  input=$1/input
  output=$1/output
  result=$1/result
  java -jar Justifier.jar "$2" <"$input" >"$result"
  diff -q "$result" "$output"
  if [ $? -eq 0 ]; then
    echo -e "\e[0;92mTest $1 passed!\e[0m"
    rm "$result"
  else
    echo -e "\e[0;91mTest $1 failed!\e[0m"
  fi
  echo
}

for file in dependencies/justifier/tests/xtrains/test*; do
  run_test "$file" "ontologies/XTRAINS.owl"
done