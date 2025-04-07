# HoneyFL
>>>>>>> 3dc857a9d50fe4cfea45b6478e8b3982a0caeb05
HoneyFL/
├── honeyfl/
│   ├── data/           # Data loading and processing
│   ├── models/         # Model architectures
│   ├── utils/          # Utility functions
│   └── main_fed.py     # Main training script
└── README.md

## Quick Start
To quickly run HoneyFL, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HoneyFL.git
   ```
2. Navigate to the project directory:
   ```bash
   cd HoneyFL
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main training script:
   ```bash
   python honeyfl/main_fed.py

## Features in Detail

### Honeypot Mechanism
HoneyFL implements a novel honeypot-based defense mechanism that:
- Injects specially crafted honeypot samples into selected clients
- Creates unique mapping patterns for attack detection
- Monitors client behavior through honeypot triggers

### Attack Detection
- Real-time monitoring of client behavior
- Detection of poisoning attacks through honeypot responses
- Calculation of Detection Success Rate (DSR) and False Positive Rate (FPR)

## Evaluation Metrics
- Model Accuracy
- Detection Success Rate (DSR)
- False Positive Rate (FPR)
=======