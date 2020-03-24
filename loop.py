#!/bin/bash

for ((i=10; i<=10000; i+=10)); 
do 
    echo -n $i','  >> file.csv
    ./run.sh $i $i $i > temp.log
    echo -n `grep 'mkl sgemm:' temp.log | awk '{ print $3 }'`',' >> file.csv
    echo -n `grep 'mkldnn sgemm:' temp.log | awk '{ print $3 }'`',' >> file.csv
    grep 'mkldnn bgemm+omp_cvt:' temp.log | awk '{ print $4 }' | sed 's/+//' | sed 's/X//' >> file.csv
done
