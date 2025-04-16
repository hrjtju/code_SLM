
for file in `ls $1`
do
    echo "Testing $1$file"
    sh ./test_flatdqn_result.sh $1$file
done