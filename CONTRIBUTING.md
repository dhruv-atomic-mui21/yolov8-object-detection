# Contributing to YOLOv8 Object Detection

Thank you for considering contributing to this project! Here's how you can help.

## 🐛 Bug Reports

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

## 💡 Feature Requests

Open an issue with:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach

## 🔧 Pull Requests

### Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/yolov8-object-detection.git
cd yolov8-object-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov  # for testing
```

### Development Workflow

1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow the code style
3. **Test**: `pytest tests/ -v`
4. **Commit**: Use clear commit messages
5. **Push**: `git push origin feature/your-feature`
6. **PR**: Open a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Keep functions focused and small

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update type hints

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
