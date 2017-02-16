@echo on
:::: 14-02-17 Tue., eval with new displacement maps
::python geonet_eval.py --eval_dir=eval/new_disp2 --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=1 --max_images=1 --num_epoch=1 --image_width=1024 --image_height=1024 --file_list=test.txt --transform=False --min_scale=1.0 :: 36450? overfit..
::python geonet_eval.py --eval_dir=eval/new_disp3 --pretrained_model_checkpoint_path=log/new_disp3/geonet.ckpt --data_dir=data/displacement --batch_size=1 --max_images=1 --num_epoch=1 --image_width=1024 --image_height=1024 --file_list=test.txt --transform=False --min_scale=1.0 :: 36497? overfit..
::python geonet_eval.py --eval_dir=eval/new_disp2_patch --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=4 --max_images=4 --num_epoch=50 --image_width=256 --image_height=256 --file_list=test.txt --transform=True --min_scale=0.03125 :: 7799? overfit.. it is supposed to be around 100
::python geonet_eval.py --eval_dir=eval/new_disp2_train_patch --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=4 --max_images=4 --num_epoch=50 --image_width=256 --image_height=256 --file_list=train.txt --transform=True --min_scale=0.03125 :: 90. seems fine.
::python geonet_eval.py --eval_dir=eval/new_disp2_train --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=1 --max_images=1 --num_epoch=1 --image_width=1024 --image_height=1024 --file_list=train.txt --transform=False --min_scale=1.0 :: around 5000.. seems that it doesn't fit for different scale (1.0) even when divided by image area (1k^2 vs. 256^2)
::python geonet_eval.py --eval_dir=eval/new_disp2_resize --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=1 --max_images=1 --num_epoch=1 --image_width=256 --image_height=256 --file_list=test.txt --transform=False --min_scale=1.0 :: 2238.. much higher than 100.. overfit..
::python geonet_eval.py --eval_dir=eval/new_disp2_train_resize --pretrained_model_checkpoint_path=log/new_disp2/geonet.ckpt --data_dir=data/displacement --batch_size=1 --max_images=1 --num_epoch=1 --image_width=256 --image_height=256 --file_list=train.txt --transform=False --min_scale=1.0 :: 334.151.. much higher than 100. might need to separate learning for each level

:::: 08-02-17 wed, first-eval
::python geonet_eval.py --eval_dir=eval/gc_noise --pretrained_model_checkpoint_path=log/gc_noise/geonet.ckpt --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --num_epoch=1

:::: 08-02-17 wed, first-eval
::python geonet_eval.py --eval_dir=eval/gc_noise --pretrained_model_checkpoint_path=log/gc_noise/geonet.ckpt --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --num_epoch=1
