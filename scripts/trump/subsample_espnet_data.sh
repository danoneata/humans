# Example:
# bash scripts/trump/subsample_espnet_data.sh trump-chunks-manual-shots
dset=$1

if [ -z $dset ]; then
    echo "ERR: Dataset is not specified"
    exit 1
fi

for s in 0 1 2 3 4; do
    for n in 30 60 120 240; do
        python scripts/subsample_espnet_data.py -d $dset -n $n --seed $s
    done
done
