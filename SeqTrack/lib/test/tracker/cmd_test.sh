epoch=(6 7 8 9 10)

for e in "${epoch[@]}"
do
    echo "run epoch:"$e
    python prompt_seqtrack.py --epoch $e
done