python train.py --vocab_file ./mimicdata/mimic3/vocab.csv \
                --code_file ./mimicdata/mimic3/TOP_50_CODES.csv \
                --embed_file ./mimicdata/mimic3/processed_full.embed \
                --description_file ./mimicdata/mimic3/description_vectors.vocab \
                --train_file ./mimicdata/mimic3/train_50.csv \
                --dev_file ./mimicdata/mimic3/dev_50.csv \
                --test_file ./mimicdata/mimic3/test_50.csv \
                --save_dir caml \
                --batch_size 32 \
                --gpus 0 \
                --epochs 200 \
                --lmbda 0 \
                --num_filter_maps 150 \
                --word_kernel_sizes 10 \
                --label_kernel_sizes 11,9,7,5 \
                --use_desc 1 \
                --nw 0 \
                --lr 0.0001 \
                --log_frq 10 \
                --method caml
