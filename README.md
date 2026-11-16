\# Driver Behaviour Detection



Real-time driver behaviour detection using a custom YOLO model.



\## Classes

\- eye close

\- eye close glass

\- eye open

\- eye open glass

\- phone use

\- yawn



\## Project parts

\- `desktop/` -> original Python/OpenCV local version

\- `model/best.onnx` -> web deployment model

\- `index.html`, `style.css`, `app.js` -> browser-based live demo



\## Local Python version

```bash

cd desktop

python -m venv venv

venv\\Scripts\\activate

pip install -r ..\\requirements.txt

python yolo\_detect.py --model best.pt --source usb0



