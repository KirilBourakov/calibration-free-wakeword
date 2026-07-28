from DiscreteMCI.DiscreteClassifier import DiscreteClassifier
from mci_wake.data import filter_training, load_raw_data

def get_models() -> tuple[DiscreteClassifier, ...]:
    model1 = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_2\checkpoints\best-model-epoch=05-val_acc=0.98.ckpt"
    )
    model2 = DiscreteLightningModule.load_from_checkpoint(
        r"D:\Coding\calibration-free-wakeword\mci_wake\scripts\lightning_logs\version_3\checkpoints\best-model-epoch=09-val_acc=0.99.ckpt"
    )
    return model1.internals, model2.internals

def main():
    models = get_models()
    emg_data_all, labels_all, subject_ids_all, adl_data, adl_ids =  filter_training(*load_raw_data(), *[m.config.customers for m in models])


if __name__ == "__main__":
    main()