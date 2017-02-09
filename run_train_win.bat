@echo on
:::: 09-02-17 thu, batch test
python geonet_train.py --log_dir=log/test --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=10 --initial_learning_rate=0.01 --decay_steps=30000

:::: 08-02-17 wed, first-train -> enough with 40k (noise intensity: 0.05, ilr 0.01)
::python geonet_train.py --log_dir=log/gc_noise --data_dir=data/dataset2 --batch_size=64 --image_width=32 --image_height=32 --max_steps=100000 --initial_learning_rate=0.01 --decay_steps=30000



