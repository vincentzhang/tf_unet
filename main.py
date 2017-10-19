from tf_unet import unet, util, image_util

#preparing data loading
data_provider = image_util.HDF5DataProvider("/data/dataset/hip/abhi")
output_path = 'output'

#setup & training
net = unet.Unet(layers=5, features_root=64, channels=1, n_class=2,
                    cost_kwargs=dict(regularizer=0.0005))
# momentum + SGD
trainer = unet.Trainer(net, optimizer="momentum",
                opt_kwargs=dict(learning_rate=0.01, momentum=0.9, decay_rate=0.95))
# adam
#trainer = unet.Trainer(net, norm_grads=True, optimizer="adam")
path = trainer.train(data_provider, output_path, training_iters=4586, epochs=2,
                dropout=0.5)

