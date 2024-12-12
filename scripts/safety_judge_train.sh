cd src\dataset\judges\safety_judge_llm
!python train.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B" \
    --train_data_name_or_path PKU-Alignment/BeaverTails:330k_train \
    --eval_data_name_or_path PKU-Alignment/BeaverTails:330k_test \
    --output_dir output/qwen2.5_1.5B_judge \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1