#! /bin/bash

# Download data
base="https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-scikit-learn-iris-basic-exercise"  
vizbase='https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-scikit-learn-iris-visualization-exercise'
knnbase='https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise'
logbase='https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-scikit-learn-logistic-regression-exercise'

for i in {1..10}; do wget  ${base}-${i}.php ; done;
for i in {1..19}; do  wget  ${vizbase}-${i}.php ; done;
for i in {1..8}; do  wget  ${knnbase}-${i}.php ; done;
for i in {1..3}; do  wget  ${logbase}-${i}.php ; done;

prefixes=`ls *php |
 grep -o  '.*exercise'  | 
 sort -u`

# Extract titles, subtitles, and code section
for prefix in ${prefixes};
do
	title=`echo $prefix | sed -e "s:python-machine-learning-::g" -e "s:scikit-learn-::g"`

	echo '# ' $title
	echo

	for i in {1..19};
	do
		f=${prefix}-${i}.php
		if [[ -f "$f" ]];
		then
			subtitle=`grep -h '^<h1'  $f | 
			sed  -e "s:<h1[^>]*>::g" \
			-e "s:</h1>::g" \
			-e "s/Python Scikit[- ]learn: //g" `

			echo '## ' $subtitle
			echo '``` python'
			sed -n '/<p><strong>Python Code:<\/strong><\/p>/,/<\/pre>/p'  $f | grep -v '^<' ;
			echo '```'
			echo 
			echo 
		fi
	done
	echo
done 