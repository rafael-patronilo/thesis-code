STUDIES=(2 3 4)

for STUDY in $STUDIES; do
    echo "Study: $STUDY"
    ./docker-run.sh IMG python src/run_script.py eval_xtrains_autoencoder L64 xtrains_autoencoders_$STUDY
done