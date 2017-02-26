# 24-02-17 Fri.
# python geonet_run.py --result_dir=result/face_whole_0.01_128 --pretrained_model_checkpoint_path=log/face_whole_0.01_128/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01
python geonet_run.py --result_dir=result/face_whole_0.01_64  --pretrained_model_checkpoint_path=log/face_whole_0.01_64/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01
python geonet_run.py --result_dir=result/face_whole_0.01_32  --pretrained_model_checkpoint_path=log/face_whole_0.01_32/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01

python geonet_run.py --result_dir=result/face_whole_0.005_128 --pretrained_model_checkpoint_path=log/face_whole_0.005_128/geonet.ckpt --data_dir=data/10FacialModels_whole --noise_level=0.005
python geonet_run.py --result_dir=result/face_whole_0.005_64  --pretrained_model_checkpoint_path=log/face_whole_0.005_64/geonet.ckpt  --data_dir=data/10FacialModels_whole --noise_level=0.005
python geonet_run.py --result_dir=result/face_whole_0.005_32  --pretrained_model_checkpoint_path=log/face_whole_0.005_32/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.005