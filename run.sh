python geonet_run.py --result_dir=result/downupflat_256_2    --checkpoint_dir=log/downupflat_256_2 --data_dir=data/sketch --moving_avg_decay=0.0
python geonet_run.py --result_dir=result/downupflat_256_2_mv --checkpoint_dir=log/downupflat_256_2 --data_dir=data/sketch --moving_avg_decay=0.9999

# python geonet_run.py --result_dir=result/downup_256    --checkpoint_dir=log/downup_256 --data_dir=data/sketch --moving_avg_decay=0.0
# python geonet_run.py --result_dir=result/downup_256_mv --checkpoint_dir=log/downup_256 --data_dir=data/sketch --moving_avg_decay=0.9999

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