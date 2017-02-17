@echo on
:::: 17-02-17 Fri., train on 128 fixed size patch, 10 faces, whole gaussian noise
python geonet_train.py --log_dir=log/face_whole_128_cont --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=50000 --transform=True --noise_level=0.005 --weight_on=False --pretrained_model_checkpoint_path=log/face_whole_128
::python geonet_train.py --log_dir=log/face_whole_128 --data_dir=data/10FacialModels_whole --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=50000 --transform=True --noise_level=0.005 --weight_on=False

:::: 16-02-17 Thu., train on 128 fixed size patch, 10 faces
::python geonet_train.py --log_dir=log/face_128 --data_dir=data/10FacialModels --file_list=train_mat.txt --batch_size=16 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --transform=True --noise_level=0.005 --weight_on=False

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:::: 14-02-17 Tue., train on 128 fixed size patch
::python geonet_train.py --log_dir=log/disp_128 --data_dir=data/displacement --batch_size=16 --image_width=128 --image_height=128 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=20000 --transform=True --min_scale=0.125 --noise_level=0.001 --weight_on=False
::python geonet_train.py --log_dir=log/disp_128_2 --data_dir=data/displacement --batch_size=16 --image_width=128 --image_height=128 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=40000 --transform=True --min_scale=0.125 --noise_level=0.001 --weight_on=False

:::: 14-02-17 Tue., 256->1024, need to run on tf CPU version (memory issue)
::python geonet_train.py --log_dir=log/disp_1024 --data_dir=data/displacement --batch_size=1 --image_width=1024 --image_height=1024 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=20000 --transform=True --min_scale=0.03125 --noise_level=0.001 --weight_on=False

:::: 13-02-17 Mon., new displacement map, lower initial learning rate 0.005, decay every 20000
::python geonet_train.py --log_dir=log/new_disp2 --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --initial_learning_rate=0.005 --decay_steps=20000 --min_scale=0.03125 --noise_level=0.001 --weight_on=False
::python geonet_train.py --log_dir=log/new_disp3 --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=20000 --min_scale=0.03125 --noise_level=0.001 --weight_on=False

:::: 12-02-17 Sun., new displacement map
::python geonet_train.py --log_dir=log/new_disp --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --min_scale=0.03125 --noise_level=0.001 --weight_on=False

:::: 09-02-17 thu, batch test
::python geonet_train.py --log_dir=log/test --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=10 --initial_learning_rate=0.01 --decay_steps=30000

:::: 08-02-17 wed, first-train -> enough with 40k (noise intensity: 0.05, ilr 0.01)
::python geonet_train.py --log_dir=log/gc_noise --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=30000