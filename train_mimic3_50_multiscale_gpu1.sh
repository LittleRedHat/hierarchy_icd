python train.py --vocab_file ./mimicdata/mimic3/vocab.csv \
                --code_file ./mimicdata/mimic3/TOP_50_CODES.csv \
                --embed_file ./mimicdata/mimic3/processed_full.embed \
                --description_file ./mimicdata/mimic3/description_vectors.vocab \
                --train_file ./mimicdata/mimic3/train_50.csv \
                --dev_file ./mimicdata/mimic3/dev_50.csv \
                --test_file ./mimicdata/mimic3/test_50.csv \
                --save_dir multiscale-gpu1 \
                --batch_size 32 \
                --num_layers 7 \
                --drop_rate 0.0 \
                --gpus 1 \
                --epochs 200 \
                --lmbda 0 \
                --num_filter_maps 200 \
                --nw 4 \
                --log_frq 10 \
                --max_length 4000 \
                --optim sgdmom \
                --lr 0.01 \
                --optim_alpha 0.9 \
                --optim_beta 0.999 \
                --milestones 10,20,30 \
                --method multiscale
