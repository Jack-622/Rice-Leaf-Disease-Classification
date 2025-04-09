import gradio as gr
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from gradio.flagging import CSVLogger

# loading the tensorflow model
stage1_model = tf.keras.models.load_model("C:/Users/DELL/Desktop/Final Project   2025-1-1/Final Project   2025-1-1/Final Project/code storage/stage_1_model-MobileNet.weights.h5")
stage2_model = tf.keras.models.load_model("C:/Users/DELL/Desktop/Final Project   2025-1-1/Final Project   2025-1-1/Final Project/code storage/Stage_2_model-MobileNet.weights.h5")

# class labels
labels = ["Bacterialblight", "Blast", "Brownspot", "Health","Tungro", "Hispa"]

# define the prediction function
def classify_image(image):
    # resize the image to the model input size
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # stage 1 prediction
    stage1_predictions = stage1_model.predict(image_array)
    stage1_confidence = stage1_predictions[0][0]

    if stage1_confidence > 0.5:
        stage2_predictions = stage2_model.predict(image_array)
        stage2_predicted_class = labels[np.argmax(stage2_predictions)]
        stage2_confidence = np.max(stage2_predictions)
        stage2_all_confidences = {label: float(conf) for label, conf in zip(labels, stage2_predictions[0])}

        all_probs_text = "\n".join([f"{label}: {conf:.2f}" for label, conf in stage2_all_confidences.items()])

        text_output = (
            f"Stage 1: Yes, Confidence: {stage1_confidence:.2f}\n"
            f"Stage 2: Predicted Class: {stage2_predicted_class}, Confidence: {stage2_confidence:.2f}\n\n"
            f"All Class Probabilities:\n{all_probs_text}"
        )

        # 画条形图
        plt.figure(figsize=(8, 4))
        plt.bar(stage2_all_confidences.keys(), stage2_all_confidences.values())
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Stage 2: Class Probabilities")
        plt.ylim(0, 1.1)

        # 保存图像为 PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        bar_chart = PILImage.open(buf)
        plt.close()

        return text_output, bar_chart
    else:
        text_output = f"Stage 1: No, Confidence: {1 - stage1_confidence:.2f}\n" \
                      "Stage 2: Not performed due to Stage 1 classification being 'No', please identify if the given image contain the rice leaf."
        return text_output, None

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(), gr.Image(type="pil")],
    title="Rice Leaf Disease Classification Terminal",
    description="This is a terminal for rice leaf disease classification. Upload an image of a rice leaf and the model will classify it!",
)


# Launch the Gradio application
demo.launch(share=True)