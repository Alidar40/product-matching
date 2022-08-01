import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data_processing.utils import get_dataloaders
from models.encoder import Encoder
from models.lit import LitSiamese
from utils import seed_everything
from models.embedder import Embedder
from config import config


if __name__ == "__main__":
    SEED = config["seed"]
    MODEL = config["model"]
    EMBEDDER = config["embedder"]
    USE_TFIDF = config["use_tfidf"]
    WANDB_ARGS = config["wandb"]
    CONTINUE_FROM_CKPT = config["continue_from_ckpt"]
    CKPT_PATH = config["ckpt_path"]
    EPOCHS = config["epochs"]
    LOG_EVERY_N_STEP = config["log_every_n_step"]
    VAL_CHECK_INTERVAL = config["val_check_interval"]
    TRAIN_DATA_PATH = config["train_data_path"]

    seed_everything(SEED)

    embedder = Embedder(EMBEDDER, USE_TFIDF)
    encoder = Encoder(embedder.emb_size)
    model = LitSiamese(encoder)
    model.train()

    train_dataloader, test_dataloader = get_dataloaders(TRAIN_DATA_PATH, embedder)

    wandb_logger = WandbLogger(project=WANDB_ARGS["project"], name=WANDB_ARGS["name"], mode=WANDB_ARGS["mode"])

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename=MODEL+"-{epoch:02d}-{val_loss:.2f}",
        dirpath=f"./checkpoints/{wandb_logger.experiment.id[-8:]}/"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=wandb_logger,
        log_every_n_steps=LOG_EVERY_N_STEP,
        callbacks=[checkpoint_callback],
        gpus=1,
        accelerator="gpu",
        devices=1,
    )

    if CONTINUE_FROM_CKPT:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=CKPT_PATH)
    else:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

