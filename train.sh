docker run -d -it\
  --name EvalGAN \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/data/,target=/opt1/data/ \
  evalgan --epochs 5 --config_path "tcn_tcn_sinus_256.json" --wandb_projekt "DockerDebug" --datapath "../data/preprocessed-data/periodic-waves/" --num_frequencies 1
