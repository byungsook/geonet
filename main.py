from config import get_config
from utils import prepare_dirs_and_logger, save_config
from trainer import Trainer
from data import BatchManager

def main(config):
    prepare_dirs_and_logger(config)    
    batch_manager = BatchManager(config)
    trainer = Trainer(config, batch_manager)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
