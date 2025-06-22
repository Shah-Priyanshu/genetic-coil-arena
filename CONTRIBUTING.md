# Contributing to Genetic Coil Arena

We welcome contributions to Genetic Coil Arena! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/genetic-coil-arena.git
   cd genetic-coil-arena
   ```

2. **Set up development environment**
   ```bash
   make setup
   # or manually:
   pip install -e ".[dev,ui,analysis]"
   pre-commit install
   ```

3. **Run tests to verify setup**
   ```bash
   make test
   pytest tests/ -v
   ```

## Code Style

- **Python**: We use `black` for formatting and `ruff` for linting
- **Type hints**: All functions should have type annotations
- **Docstrings**: Use Google-style docstrings for all public functions
- **Line length**: 88 characters (black default)

Run formatting and linting:
```bash
make format
make lint
```

## Testing

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Property tests**: Use `hypothesis` for property-based testing
- **Benchmarks**: Performance tests for critical paths

Run the test suite:
```bash
# All tests
make test

# Fast tests only
make test-fast

# Benchmarks
make benchmark
```

## Areas for Contribution

### 🚀 High Priority
- **Performance optimization**: Numba improvements, better parallelization
- **Visualization**: Enhanced Pygame renderer, web interface
- **Documentation**: Tutorials, examples, API improvements
- **Testing**: Increase coverage, add edge cases

### 🧬 Algorithm Improvements
- **Selection methods**: New parent selection strategies
- **Crossover operators**: Alternative recombination methods
- **Mutation strategies**: Adaptive mutation rates, directed mutations
- **Fitness functions**: Multi-objective optimization, novelty search

### 🎮 Game Enhancements
- **Multi-agent**: Cooperative/competitive evolution
- **Environment variations**: Different board shapes, obstacles
- **Agent architectures**: LSTM, attention mechanisms, NEAT

### 🔬 Research Features
- **Experiment tracking**: MLflow integration
- **Hyperparameter optimization**: Optuna integration
- **Analysis tools**: Better visualization, statistical tests
- **Reproducibility**: Enhanced seed management, deterministic execution

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow code style guidelines
   - Ensure all tests pass

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## Commit Message Guidelines

Use conventional commits format:
- `feat: add new selection method`
- `fix: resolve memory leak in evaluator`
- `docs: update API documentation`
- `test: add integration tests for genome`
- `perf: optimize game step function`
- `refactor: simplify agent architecture`

## Code Review Process

All contributions go through code review:

1. **Automated checks**: CI runs tests, linting, type checking
2. **Performance checks**: Benchmarks ensure no regressions
3. **Manual review**: Core maintainers review code quality
4. **Documentation**: Ensure changes are properly documented

## Performance Considerations

When contributing performance-critical code:

- **Profile first**: Use `cProfile` or `line_profiler` to identify bottlenecks
- **Benchmark**: Add performance tests for critical paths
- **Numba-compatible**: Ensure hot paths can be JIT compiled
- **Memory efficient**: Consider memory usage, especially in tight loops

## Documentation

- **API docs**: All public functions need docstrings
- **Examples**: Add usage examples for new features
- **Architecture**: Update design docs for significant changes
- **README**: Update if adding new functionality

## Issue Reporting

When reporting issues:

1. **Search existing issues** first
2. **Use issue templates** provided
3. **Include minimal reproduction** case
4. **Provide system information** (OS, Python version, etc.)
5. **Include relevant logs** or error messages

## Questions?

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Discord**: Real-time chat (link in README)

## Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: All contributors listed
- **Release notes**: Major contributions highlighted
- **Hall of Fame**: Top contributors featured

---

Thank you for contributing to Genetic Coil Arena! 🧬🐍
