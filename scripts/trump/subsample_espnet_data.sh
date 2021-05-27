for n in 30 60 120 240; do
    python scripts/subsample_espnet_data.py -d trump-chunks-cpac -n $n
done
