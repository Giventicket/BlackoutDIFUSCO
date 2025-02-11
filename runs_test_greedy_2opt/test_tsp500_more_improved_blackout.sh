export CUDA_VISIBLE_DEVICES=0

python3 -u difusco/train.py \
  --task "tsp" \
  --logger_name "tsp_diffusion_graph_tsp500_test" \
  --diffusion_type "blackout" \
  --inference_generation_type "binomial" \
  --diffusion_schedule "more_improved" \
  --do_test \
  --do_valid_only \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --training_split "./difusco/tsp500_train_concorde.txt" \
  --validation_split "./difusco/tsp500_test_concorde.txt" \
  --test_split "./difusco/tsp50_test_concorde.txt" \
  --batch_size 1 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "more_improved" \
  --inference_diffusion_steps 50 \
  --ckpt_path "/workspace/BlackoutDIFUSCO/lightning_logs/version_2/checkpoints/last.ckpt" \
  --num_states 2 \
  --alpha 0.2