from model import build_model
from utils import get_data_generators

def main():
    train_dir = "data/train"
    val_dir = "data/val"

    train_gen, val_gen = get_data_generators(train_dir, val_dir)

    model = build_model(num_classes=train_gen.num_classes)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save("model.h5")

if __name__ == "__main__":
    main()
