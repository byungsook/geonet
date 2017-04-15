python geonet_run.py --result_dir=result/100k_128/n1 --checkpoint_dir=log/100k_128/n1 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n1 --model=1
python geonet_run.py --result_dir=result/100k_128/n2 --checkpoint_dir=log/100k_128/n2 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n2 --model=1
python geonet_run.py --result_dir=result/100k_128/n3 --checkpoint_dir=log/100k_128/n3 --data_dir=data/faces_low_res/maps/100k/original --noise_level=n3 --model=1

# 09-03-17. Thu. Test with weights
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.00 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.00/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.05 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.05/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.10 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.10/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.25 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.25/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.50 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.50/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/weight/face_whole_0.010_128_w0.75 --pretrained_model_checkpoint_path=log/weight/face_whole_0.010_128_w0.75/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01


# python geonet_run.py --result_dir=result/face_whole_0.01_512 --pretrained_model_checkpoint_path=log/face_whole_0.01_512/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/face_whole_0.01_256 --pretrained_model_checkpoint_path=log/face_whole_0.01_256/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.01

# python geonet_run.py --result_dir=result/face_whole_0.005_512 --pretrained_model_checkpoint_path=log/face_whole_0.005_512_cont/geonet.ckpt-95000 --data_dir=data/10FacialModels_whole --noise_level=0.005
# python geonet_run.py --result_dir=result/face_whole_0.005_256 --pretrained_model_checkpoint_path=log/face_whole_0.005_256/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.005

# 24-02-17 Fri.
# python geonet_run.py --result_dir=result/face_whole_0.01_128 --pretrained_model_checkpoint_path=log/face_whole_0.01_128/geonet.ckpt --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/face_whole_0.01_64  --pretrained_model_checkpoint_path=log/face_whole_0.01_64/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01
# python geonet_run.py --result_dir=result/face_whole_0.01_32  --pretrained_model_checkpoint_path=log/face_whole_0.01_32/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.01

# python geonet_run.py --result_dir=result/face_whole_0.005_128 --pretrained_model_checkpoint_path=log/face_whole_0.005_128/geonet.ckpt-100000 --data_dir=data/10FacialModels_whole --noise_level=0.005
# python geonet_run.py --result_dir=result/face_whole_0.005_64  --pretrained_model_checkpoint_path=log/face_whole_0.005_64/geonet.ckpt  --data_dir=data/10FacialModels_whole --noise_level=0.005
# python geonet_run.py --result_dir=result/face_whole_0.005_32  --pretrained_model_checkpoint_path=log/face_whole_0.005_32/geonet.ckpt-100000  --data_dir=data/10FacialModels_whole --noise_level=0.005