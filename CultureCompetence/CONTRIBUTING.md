# Contributing to CultureAwarenessTest

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CultureAwarenessTest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For development with GPU support, ensure CUDA compatibility with TensorFlow 2.10.1

## Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Include docstrings for all public methods
- Use meaningful variable names

## Testing

- Test models on both lamp and carpet datasets
- Validate bias mitigation effectiveness
- Check robustness against adversarial attacks
- Verify discriminator accuracy

## Adding New Models

1. Extend `GeneralModelClass` for new model types
2. Implement required methods: `__call__`, `test`, `get_model_stats`
3. Add model-specific training logic
4. Update `ProcessingClass` to support the new model

## Adding New Datasets

1. Update `Utils/Data/deep_paths.py` or `shallow_paths.py`
2. Ensure culture labels are properly encoded
3. Test data loading with `DataClass`

## Reporting Issues

When reporting bugs or requesting features:
- Include dataset used (lamps/carpets)
- Specify model configuration (standard/mitigated)
- Provide error messages and stack traces
- Include system information (OS, GPU, TensorFlow version)

## Research Contributions

This is a research project. Contributions should:
- Address cultural bias in ML models
- Provide empirical evidence of effectiveness
- Include proper evaluation metrics
- Document methodology and results

## License

By contributing, you agree to license your contributions under the same license as the project (see LICENSE file).