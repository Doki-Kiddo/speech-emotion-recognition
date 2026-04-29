from tensorflow.keras import layers, models, regularizers

from src.config import MAX_FRAMES, N_MELS


def build_cnn(num_classes: int, augment: bool = False):
    augmentation_layers = []
    if augment:
        augmentation_layers = [
            layers.GaussianNoise(0.03),
            layers.RandomTranslation(height_factor=0.04, width_factor=0.08, fill_mode="constant"),
            layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.08, 0.08), fill_mode="constant"),
        ]

    model = models.Sequential(
        [
            layers.Input(shape=(N_MELS, MAX_FRAMES, 1)),
            *augmentation_layers,
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.3),
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
