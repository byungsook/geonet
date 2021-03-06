python geonet_train.py --log_dir=log/100k_128/n1 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n1 --file_list=train_mat.txt --batch_size=8 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --transform=True --model=1
python geonet_train.py --log_dir=log/100k_128/n2 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n2 --file_list=train_mat.txt --batch_size=8 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --transform=True --model=1
python geonet_train.py --log_dir=log/100k_128/n3 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n3 --file_list=train_mat.txt --batch_size=8 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --transform=True ----model=1

###########################

# python geonet_train.py --log_dir=log/face_whole_0.010_128_d2 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=8 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False --model=2

# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.10_wr3 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.10 --weight_ratio=3.0

# 03-03-17 Fri. train on weight sigma 0.5, 0.75 noise 0.01, patch 128 with batch 16, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.5  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.5
# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.75 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.75
# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.25 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.25
# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.10_cont --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.10 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.10/geonet.ckpt-65000
# python geonet_train.py --log_dir=log/weight/face_whole_0.010_128_w0.05 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=True --weight_sigma=0.05

# python geonet_train.py --log_dir=log/face_whole_0.005_512_cont  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=1 --image_width=512 --image_height=512 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False --pretrained_model_checkpoint_path=log/face_whole_0.005_512/geonet.ckpt-90000

# 27-02-17 Mon., train on noise 0.010, 0.005 patch size 256 with batch 4, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/face_whole_0.01_256  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=4 --image_width=256 --image_height=256 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False
# python geonet_train.py --log_dir=log/face_whole_0.005_256 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=4 --image_width=256 --image_height=256 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False

# 26-02-17 Sun., re-train on noise 0.005 patch size 128 with batch 16, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/face_whole_0.005_128_2 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False --pretrained_model_checkpoint_path=log/face_whole_0.005_128/geonet.ckpt

# # 26-02-17 Sun., re-train on noise 0.010 patch size 128 with batch 16, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/face_whole_0.01_128_2 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False

# 24-02-17 Fri., re-train on noise 0.005 patch size 32 with batch 16, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/face_whole_32_2  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=32 --image_height=32   --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False

# 21-02-17 Tue., train on noise 0.01, patch size 128, 64, 32 with batch 16, 10 faces, whole gaussian noise
# python geonet_train.py --log_dir=log/face_whole_0.01_128 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False
# python geonet_train.py --log_dir=log/face_whole_0.01_64  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=64 --image_height=64   --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False
# python geonet_train.py --log_dir=log/face_whole_0.01_32  --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=32 --image_height=32   --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.010 --weight_on=False

# 20-02-17 Mon., train on 64 fixed size patch, batch 16, 10 faces, whole gaussian noise
#python geonet_train.py --log_dir=log/face_whole_64 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=64 --image_height=64 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False

# 19-02-17 Sun., train on 64 fixed size patch, 10 faces, whole gaussian noise
#python geonet_train.py --log_dir=log/face_whole_64 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=64 --image_width=64 --image_height=64 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False

# 18-02-17 Sat., train on 128 fixed size patch, 10 faces, whole gaussian noise, continued
#python geonet_train.py --log_dir=log/face_whole_128_cont --data_dir=data/10FacialModels_whole  --pretrained_model_checkpoint_path=log/face_whole_128/geonet.ckpt --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --noise_level=0.005 --weight_on=False

# 17-02-17 Fri., train on 128 fixed size patch, 10 faces, 0.01!
#python geonet_train.py --log_dir=log/face_128 --data_dir=data/10FacialModels --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=50000 --transform=True --noise_level=0.005 --weight_on=False --pretrained_model_checkpoint_path=log/face_128_0.01/geonet.ckpt

# 16-02-17 Thu., train on 128 fixed size patch, 10 faces
#python geonet_train.py --log_dir=log/face_128 --data_dir=data/10FacialModels --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.005 --decay_steps=30000 --transform=True --noise_level=0.005 --weight_on=False

# 09-02-17 thu., train with dp. map
#python geonet_train.py --log_dir=log/disp --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --decay_steps=30000
