# Sign Language Translator

A real-time sign language alphabet recognition application built with Python, OpenCV, and PyTorch. The program reads frames from a webcam, classifies the hand gesture inside a defined region, builds the detected letters into text, and can read the final text aloud using text-to-speech.

The current model supports the English alphabet (`A-Z`) and three control labels: `SPACE`, `DEL`, and `NOTHING`.

## Features

- Real-time webcam input
- CNN-based gesture classification
- Configurable region of interest (ROI)
- Temporal filtering to reduce unstable predictions
- Text construction using letter and command labels
- Automatic sequence finalization after inactivity
- Optional text-to-speech output
- Mock predictor for development and testing
- Unit tests for preprocessing, decoding, model loading, and text handling

## Requirements

- Python 3.11 or newer
- A working webcam
- The trained model file at `models/final_model.pth`

Text-to-speech also requires a speech engine supported by `pyttsx3` on the operating system.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/FDamirchi/sign-language-translator.git
cd sign-language-translator
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Or on Linux and macOS:

```bash
source .venv/bin/activate
```

Install the project with the model and text-to-speech dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[model,tts]"
```

## Running the Application

Start the program from the repository root:

```bash
python main.py
```

Place your hand inside the region shown on the camera window and hold the gesture until it is accepted. Use the command gestures to add a space or delete the previous character.

Press `Q` or `Esc` to close the application.

## How It Works

1. OpenCV captures a frame from the webcam.
2. The configured region of interest is cropped from the frame.
3. The image is resized, normalized, and passed to the PyTorch model.
4. A temporal decoder filters short or low-confidence predictions.
5. Accepted labels are added to the text buffer.
6. After a period of inactivity, the sequence is finalized and sent to the speech engine.

The prediction, decoder state, current text, and most recently finalized text are displayed on the camera window.

## Project Structure

```text
sign-language-translator/
├── models/                     # Trained model files
├── notebooks/                  # Experiments and training notebooks
├── reports/                    # Project report and related assets
├── src/sign_translator/
│   ├── decoding/               # Temporal decoding and text handling
│   ├── inference/              # Model loading and prediction
│   ├── speech/                 # Text-to-speech backends
│   ├── vision/                 # Camera, ROI, and display overlay
│   ├── app.py                  # Main application loop
│   ├── config.py               # Application configuration
│   └── labels.py               # Supported model labels
├── tests/                      # Unit tests
├── main.py                     # Application entry point
└── pyproject.toml              # Package and dependency configuration
```

## Configuration

Default settings are defined in `src/sign_translator/config.py`. The main configurable values include:

- Camera index and resolution
- ROI position and size
- Model path and input dimensions
- Prediction confidence threshold
- Gesture hold and release times
- Sequence inactivity timeout
- Predictor and speech backends

The default predictor uses `models/final_model.pth`. A mock backend is also available for testing application behavior without loading the trained model.

## Tests

Install the development dependencies:

```bash
pip install -e ".[dev,model]"
```

Run the test suite:

```bash
pytest
```

Run the linter:

```bash
ruff check .
```

## Limitations

This project recognizes a fixed set of isolated alphabet and command gestures. It does not currently interpret continuous signing, facial expressions, motion-based signs, or full sign-language grammar. Recognition quality is also affected by lighting, camera position, background, and hand placement.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
