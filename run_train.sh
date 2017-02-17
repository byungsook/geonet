# 17-02-17 Fri., train on 128 fixed size patch, 10 faces, 0.01!
python geonet_train.py --log_dir=log/face_128 --data_dir=data/10FacialModels --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=50000 --transform=True --noise_level=0.005 --weight_on=False --pretrained_model_checkpoint_path=log/face_128_0.01/geonet.ckpt

# 16-02-17 Thu., train on 128 fixed size patch, 10 faces
#python geonet_train.py --log_dir=log/face_128 --data_dir=data/10FacialModels --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.005 --decay_steps=30000 --transform=True --noise_level=0.005 --weight_on=False

# 09-02-17 thu., train with dp. map
#python geonet_train.py --log_dir=log/disp --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --decay_steps=30000
