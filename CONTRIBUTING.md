# Contributing to ReCT-VLM

Thank you for your interest in contributing to ReCT-VLM! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, GPU, etc.)
   - Error messages and stack traces

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/NEWMES-AI/ReCT-VLM.git
   cd ReCT-VLM
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   conda create -n rect-vlm-dev python=3.10
   conda activate rect-vlm-dev
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow the coding style (PEP 8)
   - Add tests for new features
   - Update documentation as needed
   - Ensure all tests pass

5. **Run tests and linting**
   ```bash
   # Run tests
   pytest tests/

   # Format code
   black rect_vlm/
   isort rect_vlm/

   # Check linting
   flake8 rect_vlm/
   mypy rect_vlm/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   # or
   git commit -m "fix: fix bug description"
   ```

   **Commit message format**:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing done
     - Screenshots (if applicable)

### Code Style Guidelines

- **Python**: Follow PEP 8
- **Docstrings**: Use Google style
- **Type hints**: Add type hints to function signatures
- **Comments**: Write clear, concise comments
- **Naming**: Use descriptive variable/function names

**Example**:
```python
def compute_dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5
) -> float:
    """
    Compute Dice coefficient between prediction and target.

    Args:
        prediction: Predicted segmentation mask (B, D, H, W)
        target: Ground truth mask (B, D, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score as a float value between 0 and 1

    Example:
        >>> pred = torch.rand(2, 64, 512, 512)
        >>> target = torch.randint(0, 2, (2, 64, 512, 512)).float()
        >>> score = compute_dice_score(pred, target)
    """
    pred = prediction.contiguous().view(-1)
    tgt = target.contiguous().view(-1)

    intersection = (pred * tgt).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + tgt.sum() + smooth)

    return dice.item()
```

### Testing

- Write unit tests for new features
- Ensure all existing tests pass
- Aim for >80% code coverage
- Test on multiple Python versions (3.10, 3.11)

**Running tests**:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_vision_encoder.py

# Run with coverage
pytest --cov=rect_vlm tests/
```

### Documentation

- Update README.md for major changes
- Add docstrings to all public functions/classes
- Update relevant documentation in `docs/`
- Add examples for new features

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] No merge conflicts
- [ ] Code is properly formatted

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/NEWMES-AI/ReCT-VLM.git
cd ReCT-VLM

# Create development environment
conda create -n rect-vlm-dev python=3.10
conda activate rect-vlm-dev

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Project Structure

```
rect_vlm/
├── model/          # Model implementations
├── training/       # Training infrastructure
├── utils/          # Utility functions
└── scripts/        # Executable scripts
```

### Adding a New Feature

1. **Discuss first**: Open an issue to discuss the feature
2. **Create branch**: `git checkout -b feature/feature-name`
3. **Implement**: Write code with tests and docs
4. **Test**: Ensure all tests pass
5. **Submit PR**: Create pull request with description

### Fixing a Bug

1. **Report bug**: Create an issue with reproduction steps
2. **Create branch**: `git checkout -b fix/bug-description`
3. **Write test**: Add test that fails due to the bug
4. **Fix bug**: Implement the fix
5. **Verify**: Ensure test passes and no regressions
6. **Submit PR**: Create pull request

## Code Review Process

1. **Automatic checks**: CI/CD runs tests and linting
2. **Maintainer review**: Core team reviews code
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves PR
5. **Merge**: PR is merged to main branch

## Community Guidelines

- **Be respectful**: Treat everyone with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Maintainers review PRs as time permits
- **Be collaborative**: Work together to improve the project

## Questions?

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Email**: [contact email] for private inquiries

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in academic publications (for significant contributions)

Thank you for contributing to ReCT-VLM! 🎉
