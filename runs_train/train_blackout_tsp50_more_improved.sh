python3 -u difusco/train.py \
  --task "tsp" \
  --logger_name "diffusco" \
  --diffusion_type "blackout" \
  --inference_generation_type "binomial" \
  --diffusion_schedule "more_improved" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --training_split "/data/tsp_dataset/tsp50_train_concorde.txt" \
  --validation_split "/data/tsp_dataset/tsp50_test_concorde.txt" \
  --test_split "/data/tsp_dataset/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "more_improved" \
  --inference_diffusion_steps 50 \
  --num_states 2 \
  --alpha 0.2
