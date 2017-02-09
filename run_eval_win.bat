@echo on
rem 08-02-17 wed, first-eval
python geonet_eval.py --eval_dir=eval/gc_noise --pretrained_model_checkpoint_path=log/gc_noise/geonet.ckpt --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --num_epoch=1
