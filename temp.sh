STUDIES=(2 3 4)

for STUDY in ${STUDIES[@]}; do
    python src/run_script.py eval_xtrains_autoencoder L64 storage/studies/xtrains_autoencoders_${STUDY}
done