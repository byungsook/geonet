python geonet_eval.py --eval_dir=eval/100k_128/n1  --checkpoint_dir=log/100k_128/n1 --noise_level=n1 --data_dir=data/faces_low_res/maps/100k/original --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128  --image_height=128 --transform=True --model=1
python geonet_eval.py --eval_dir=eval/100k_128/n2  --checkpoint_dir=log/100k_128/n2 --noise_level=n2 --data_dir=data/faces_low_res/maps/100k/original --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128  --image_height=128 --transform=True --model=1
python geonet_eval.py --eval_dir=eval/100k_128/n3  --checkpoint_dir=log/100k_128/n3 --noise_level=n3 --data_dir=data/faces_low_res/maps/100k/original --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128  --image_height=128 --transform=True --model=1


# # 26-03-2017, Sun. Eval model 2, noise level 0.01 for 128 patch trained networks
# python geonet_eval.py --eval_dir=eval/face_whole_0.010_128_d2  --pretrained_model_checkpoint_path=log/face_whole_0.010_128_d2/geonet.ckpt-100000  --noise_level=0.010 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128  --image_height=128 --transform=True --weight_on=False --model=2
# python geonet_eval.py --eval_dir=eval/face_whole_0.010_128_d2_tr  --pretrained_model_checkpoint_path=log/face_whole_0.010_128_d2/geonet.ckpt-100000  --noise_level=0.010 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=128 --image_height=128 --transform=True --weight_on=False --model=2

# # 01-03-2017, Wed. Eval both noise level 0.01 and 0.005 for 512 patch trained networks
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_512  --pretrained_model_checkpoint_path=log/face_whole_0.005_512_cont/geonet.ckpt-95000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=512  --image_height=512 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_512_tr  --pretrained_model_checkpoint_path=log/face_whole_0.005_512_cont/geonet.ckpt-95000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=512  --image_height=512 --transform=True

# python geonet_eval.py --eval_dir=eval/face_whole_0.01_512  --pretrained_model_checkpoint_path=log/face_whole_0.01_512/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=512  --image_height=512 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_512_tr  --pretrained_model_checkpoint_path=log/face_whole_0.01_512/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=512 --image_height=512 --transform=True


# 01-03-2017, Wed. Eval both noise level 0.01 and 0.005 for 256 patch trained networks
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_256  --pretrained_model_checkpoint_path=log/face_whole_0.005_256/geonet.ckpt-100000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=256  --image_height=256 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_256_tr  --pretrained_model_checkpoint_path=log/face_whole_0.005_256/geonet.ckpt-100000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=256  --image_height=256 --transform=True

# python geonet_eval.py --eval_dir=eval/face_whole_0.01_256  --pretrained_model_checkpoint_path=log/face_whole_0.01_256/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=256  --image_height=256 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_256_tr  --pretrained_model_checkpoint_path=log/face_whole_0.01_256/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=256 --image_height=256 --transform=True


# 26-02-2017, Sun. Eval both noise level 0.01 and 0.005 for 32, 64 and 128 patch trained networks
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_32  --pretrained_model_checkpoint_path=log/face_whole_0.005_32/geonet.ckpt-100000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=32  --image_height=32 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_32_tr  --pretrained_model_checkpoint_path=log/face_whole_0.005_32/geonet.ckpt-100000  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=32  --image_height=32 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_64  --pretrained_model_checkpoint_path=log/face_whole_0.005_64/geonet.ckpt  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=64  --image_height=64 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_64_tr  --pretrained_model_checkpoint_path=log/face_whole_0.005_64/geonet.ckpt  --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=64  --image_height=64 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_128 --pretrained_model_checkpoint_path=log/face_whole_0.005_128/geonet.ckpt-100000 --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128 --image_height=128 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.005_128_tr --pretrained_model_checkpoint_path=log/face_whole_0.005_128/geonet.ckpt-100000 --noise_level=0.005 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=128 --image_height=128 --transform=True

# python geonet_eval.py --eval_dir=eval/face_whole_0.01_32  --pretrained_model_checkpoint_path=log/face_whole_0.01_32/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=32 --image_height=32 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_32_tr  --pretrained_model_checkpoint_path=log/face_whole_0.01_32/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=32 --image_height=32 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_64  --pretrained_model_checkpoint_path=log/face_whole_0.01_64/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=64 --image_height=64 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_64_tr  --pretrained_model_checkpoint_path=log/face_whole_0.01_64/geonet.ckpt-100000  --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=64 --image_height=64 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_128 --pretrained_model_checkpoint_path=log/face_whole_0.01_128/geonet.ckpt --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=test_mat.txt --batch_size=8 --max_images=8 --num_epoch=100 --image_width=128 --image_height=128 --transform=True
# python geonet_eval.py --eval_dir=eval/face_whole_0.01_128_tr --pretrained_model_checkpoint_path=log/face_whole_0.01_128/geonet.ckpt --noise_level=0.01 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --max_images=8 --num_epoch=10 --image_width=128 --image_height=128 --transform=True

# 22-02-17 Wed.
#python geonet_eval.py --eval_dir=eval/face_whole_0.005_128 --pretrained_model_checkpoint_path=log/face_whole_128/geonet.ckpt --data_dir=data/10FacialModels_whole --batch_size=1 --image_width=1024 --image_height=1024 --num_epoch=1
#python geonet_eval.py --eval_dir=eval/face_whole_0.005_64  --pretrained_model_checkpoint_path=log/face_whole_64/geonet.ckpt  --data_dir=data/10FacialModels_whole --batch_size=1 --image_width=1024 --image_height=1024 --num_epoch=1

# 08-02-17 wed, first-eval
#python geonet_eval.py --eval_dir=eval/test --pretrained_model_checkpoint_path=log/test/geonet.ckpt --data_dir=data/dataset1 --batch_size=64 --image_width=32 --image_height=32 --num_epoch=1
