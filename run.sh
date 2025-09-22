#!/bin/bash

filename="runs"
line_number=0

while IFS= read -r line || [[ -n "$line" ]]; do
    ((line_number++))
    $line &> test_${line_number}.log
done < "$filename"
