# TFT Bot

A Python project for TFT (Teamfight Tactics) automation.

## Project Structure

```
tft-bot/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   └── main.py           # Main application entry point
├── tests/                 # Test files
│   └── __init__.py       # Test package initialization
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python src/main.py
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Add your license here]
