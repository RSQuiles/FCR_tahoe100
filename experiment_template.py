from fcr import FCR_sim

config_path = "../config_template.json"
model = FCR_sim(config_path)
model.train_fcr()