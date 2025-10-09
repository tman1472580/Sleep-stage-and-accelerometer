# Polar H10 Real-time Sleep Stage Scoring

This system combines Web Bluetooth connectivity to a Polar H10 heart rate monitor with real-time sleep stage classification using a U-Time neural network model.

## Directory Structure

```
sleep-stage-and-accelerometer/
├── backend/
│   ├── server.py                 # Flask backend for sleep scoring
│   ├── requirements.txt          # Python dependencies
│   └── models/
│       ├── model/
│       │   └── model_weights.h5  # Trained U-Time model weights
│       └── hyperparameters/
│           └── hparams.yaml      # Model hyperparameters
└── web/
    └── 7heart-beat-polar-h10.html  # Frontend interface
```

## Setup Instructions

### 1. Backend Setup

```bash
cd sleep-stage-and-accelerometer/backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install zmax_datasets package
# Option A: If you have it locally
pip install -e /path/to/zmax_datasets

# Option B: If it's in a git repository
pip install git+https://github.com/your-repo/zmax_datasets.git

# Verify model files exist
ls -l models/model/model_weights.h5
ls -l models/hyperparameters/hparams.yaml
```

### 2. Start the Backend Server

```bash
cd sleep-stage-and-accelerometer/backend
python server.py
```

You should see:
```
[start] Starting sleep scoring server...
[start] Model path: ./models/model/model_weights.h5
[start] Epoch length: 30s
[start] Sampling rate: 200 Hz
[init] Loading model from: .../models
[init] Model loaded successfully
 * Running on http://127.0.0.1:5000
```

### 3. Open the Frontend

```bash
# Option A: Use Python's built-in server (recommended)
cd sleep-stage-and-accelerometer/web
python3 -m http.server 8000

# Then open: http://localhost:8000/7heart-beat-polar-h10.html
```

```bash
# Option B: Use any web server that supports HTTPS or localhost
# The interface requires a secure context for Web Bluetooth
```

### 4. Using the System

1. **Check Backend Status**: The frontend will show "Backend Status: ✓ Connected" if the server is running

2. **Connect Polar H10**:
   - Put on your Polar H10 chest strap
   - Click "🔗 Connect H10"
   - Select your device from the Bluetooth pairing dialog

3. **View Live Sleep Stages**:
   - Data collection starts immediately
   - After ~30 seconds, the first sleep stage prediction appears
   - Updates every 30 seconds (standard epoch length)
   - Shows confidence levels for Wake/NREM/REM

4. **Recording**:
   - Click "⏺️ Start Recording" to save data to CSV
   - CSV includes: timestamp, x, y, z, magnitude, click, **sleep_stage**
   - Click "⏹️ Stop Recording" when done

## API Endpoints

The backend server provides these endpoints:

- `GET /health` - Check if server and model are loaded
- `POST /api/samples` - Send accelerometer samples (auto-called by frontend)
- `GET /api/latest` - Get most recent sleep stage prediction
- `GET /api/history` - Get all predictions since startup
- `POST /api/reset` - Clear buffer and predictions

## Configuration

### Backend (server.py)

```python
EPOCH_LENGTH_S = 30      # Sleep scoring epoch length
SAMPLING_RATE = 200      # Polar H10 accelerometer rate
WEIGHTS_PATH = "./models/model/model_weights.h5"
```

### Frontend (HTML file)

```javascript
const BACKEND_URL = 'http://127.0.0.1:5000';
const SEND_BATCH_SIZE = 200;     // Samples per batch
const SEND_INTERVAL_MS = 1000;   // Send every 1 second
```

## Troubleshooting

### Backend won't start
- **Missing zmax_datasets**: Install the package (see Setup step 1)
- **Model not found**: Verify `models/model/model_weights.h5` exists
- **Port 5000 in use**: Change port in `server.py`: `app.run(port=5001)`

### Frontend shows "Backend Offline"
- Ensure backend server is running: `python server.py`
- Check URL matches: default is `http://127.0.0.1:5000`
- Look for CORS errors in browser console

### Bluetooth connection fails
- **Use Chrome or Edge** (Firefox doesn't support Web Bluetooth fully)
- Must use **HTTPS or localhost** (not `file://`)
- Polar H10 must be in pairing mode (wet the electrodes)
- Try refreshing the page and reconnecting

### No sleep predictions appearing
- Wait at least 30 seconds for first epoch
- Check browser console for errors
- Verify backend shows: "Model loaded successfully"
- Check `/api/latest` endpoint manually: `curl http://127.0.0.1:5000/api/latest`

### CSV file missing sleep_stage column
- Ensure backend is running BEFORE starting recording
- Check "Backend Status" shows "✓ Connected"
- Sleep stage will be empty until first prediction (after 30s)

## Data Flow

```
Polar H10 (200 Hz) 
    ↓ [Web Bluetooth]
Frontend (JavaScript)
    ↓ [HTTP POST every 1s]
Backend (Flask + U-Time)
    ↓ [Prediction every 30s]
Frontend Display + CSV File
```

## Model Requirements

The U-Time model must be trained on accelerometer data with:
- **Input**: 3-channel accelerometer (x, y, z)
- **Sampling rate**: 200 Hz
- **Epoch length**: 30 seconds (configurable)
- **Output classes**: Wake, NREM, REM (3 classes)

## Browser Compatibility

- ✅ Chrome 56+ (Windows, Mac, Linux, Android)
- ✅ Edge 79+ (Windows, Mac)
- ❌ Firefox (limited Web Bluetooth support)
- ❌ Safari (no Web Bluetooth support)

## License

Adjust according to your project's license.