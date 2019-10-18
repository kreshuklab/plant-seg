import os
import wget
import yaml
config_train = "/config_train.yml"
best_model = "/best_checkpoint.pytorch"
last_model = "/last_checkpoint.pytorch"


def check_models(model_name):
    model_dir = os.path.expanduser("~") + "/.plantseg_models/" + model_name + "/"

    if ~os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if (os.path.exists(model_dir + config_train) and
        os.path.exists(model_dir + best_model) and
        os.path.exists(model_dir + last_model)) and False:
        pass
    else:
        config = yaml.load(open("./models/models_zoo.yaml", 'r'), Loader=yaml.FullLoader)
        url = config["unet_ds2_bce"]

        wget.download(url + config_train[1:], out=model_dir)
        wget.download(url + best_model[1:], out=model_dir)
        wget.download(url + last_model[1:], out=model_dir)



    return 0
