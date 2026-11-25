
const video = document.getElementById("camera");
const canvas = document.getElementById("snapshot");
const preview = document.getElementById("preview");
const statusText = document.getElementById("status");
const startButton = document.getElementById("startCamera");
const captureButton = document.getElementById("capture");
const predictButton = document.getElementById("predict");
const results = document.getElementById("results");

let cameraStream = null;

startButton.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });

    video.srcObject = cameraStream;
    await video.play();

    captureButton.disabled = false;
    statusText.textContent = "Camera started. Stand back to show your full body.";
  } catch (error) {
    statusText.textContent =
      "Camera could not start. Check your browser's camera permission.";
    console.error(error);
  }
});

captureButton.addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) {
    statusText.textContent = "Please wait for the camera preview.";
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  canvas.getContext("2d").drawImage(
    video, 0, 0, canvas.width, canvas.height
  );

  preview.src = canvas.toDataURL("image/jpeg", 0.95);
  preview.hidden = false;
  predictButton.disabled = false;
  statusText.textContent = "Photo captured. Review it before predicting.";
});

predictButton.addEventListener("click", async () => {
  predictButton.disabled = true;
  statusText.textContent = "Running prediction...";

  try {
    const blob = await new Promise(resolve =>
      canvas.toBlob(resolve, "image/jpeg", 0.95)
    );

    if (!blob) throw new Error("Could not create the image.");

    const formData = new FormData();
    formData.append("image", blob, "capture.jpg");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    results.textContent =
      `Estimated height: ${data.height_cm} cm | ` +
      `Estimated weight: ${data.weight_kg} kg`;

    statusText.textContent =
      "Prediction complete. These are estimates, not verified measurements.";
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
});
