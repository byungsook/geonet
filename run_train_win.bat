@echo on
:::: 13-02-17 Mon., new displacement map, lower initial learning rate 0.005, decay every 20000
python geonet_train.py --log_dir=log/new_disp2 --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --initial_learning_rate=0.005 --decay_steps=20000 --min_scale=0.03125 --noise_level=0.001 --weight_on=False

:::: 12-02-17 Sun., new displacement map
::python geonet_train.py --log_dir=log/new_disp --data_dir=data/displacement --batch_size=4 --image_width=256 --image_height=256 --max_steps=50000 --initial_learning_rate=0.01 --decay_steps=30000 --min_scale=0.03125 --noise_level=0.001 --weight_on=False

:::: 09-02-17 thu, batch test
::python geonet_train.py --log_dir=log/test --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=10 --initial_learning_rate=0.01 --decay_steps=30000

:::: 08-02-17 wed, first-train -> enough with 40k (noise intensity: 0.05, ilr 0.01)
::python geonet_train.py --log_dir=log/gc_noise --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=30000