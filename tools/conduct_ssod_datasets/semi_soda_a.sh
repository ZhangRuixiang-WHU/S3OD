
OFFSET=$RANDOM
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        python tools/conduct_ssod_datasets/semi_soda_a.py --percent ${percent} --seed ${fold} --seed-offset ${OFFSET}
    done
done
