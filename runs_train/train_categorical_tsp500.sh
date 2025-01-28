python3 -u difusco/train.py \
  --task "tsp" \
  --logger_name "diffusco" \
  --diffusion_type "categorical" \
  --diffusion_schedule "cosine" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --training_split "/data/tsp_dataset/tsp500_train_concorde.txt" \
  --validation_split "/data/tsp_dataset/tsp500_test_concorde.txt" \
  --test_split "/data/tsp_dataset/tsp500_test_concorde.txt" \
  --sparse_factor 50 \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50