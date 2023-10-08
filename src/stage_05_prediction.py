import argparse
import os
import logging
import tensorflow as tf
from src.utils.common import read_yaml

STAGE = "PREDICTION"  # <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def main(config_path, image_path):
    # Read config files
    config = read_yaml(config_path)

    params = config["params"]

    # Load the trained model
    path_to_model = os.path.join(
        config["data"]["local_dir"],
        config["data"]["model_dir"],
        config["data"]["trained_model_file"])

    logging.info(f"Load the trained model from: {path_to_model}")
    classifier = tf.keras.models.load_model(path_to_model)

    # Load and preprocess the image for prediction
    logging.info("Load and preprocess the image for prediction")
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=params["img_shape"][:-1]
    )

    logging.info("Converting image to array")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    logging.info("Add batch dimension")
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    logging.info("Normalize pixel values to [0, 1]")
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = classifier.predict(img_array)
    logging.info(f"Predictions shape: {predictions.shape}")

    # Get class labels from the model
    class_labels = config["data"]["class_labels"]
    logging.info(f"Class labels: {class_labels}")

    # Display the predicted class and probability
    predicted_class = class_labels[predictions.argmax()]
    logging.info(f"Predicted class: {predicted_class}")
    predicted_probability = predictions.max()
    logging.info(f"Predicted probability: {predicted_probability}")

    print(f"Predicted Class: {predicted_class}")
    print(f"Predicted Probability: {predicted_probability:.2%}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--image", "-i", required=True, help="Path to the image for prediction")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,image_path=parsed_args.image)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
