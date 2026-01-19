# Project Documentation Index

This document indexes all documentation and resources available in the stochlib project.

## 📚 Core Documentation

### [README.md](README.md)
Main project overview, installation, and quick start guide.

### [FILE_STRUCTURE.md](FILE_STRUCTURE.md)
Complete architectural documentation with:
- Hierarchical project structure
- Module descriptions and purposes
- Public API reference
- Test structure overview
- Development standards

### [LOGGING_GUIDE.md](LOGGING_GUIDE.md)
Practical logging usage guide with:
- Quick start examples
- Configuration options
- Format levels (DEFAULT, VERBOSE, DEBUG)
- Environment variables
- Common patterns
- Troubleshooting

### [LOGGING_IMPROVEMENTS.md](LOGGING_IMPROVEMENTS.md)
Summary of logging system enhancements:
- New features (file logging, decorators, environment control)
- Files modified
- Test validation results

## 📖 Examples and Tutorials

### [examples/README.md](examples/README.md)
Comprehensive guide to example notebooks with:
- Quick start instructions
- Notebook descriptions
- Recommended learning path
- Common code patterns
- Troubleshooting tips

### Example Notebooks

| Notebook | Topic | Difficulty | Duration |
|----------|-------|------------|----------|
| [01_fokker_planck_intro.ipynb](examples/01_fokker_planck_intro.ipynb) | FP PDE solver basics | Beginner | 5-10 min |
| [02_sde_paths_intro.ipynb](examples/02_sde_paths_intro.ipynb) | SDE path simulation | Intermediate | 10-15 min |
| [03_boundary_conditions.ipynb](examples/03_boundary_conditions.ipynb) | BC types & effects | Intermediate | 10-15 min |
| [04_parameter_sweep_analysis.ipynb](examples/04_parameter_sweep_analysis.ipynb) | Convergence & stability | Advanced | 15-20 min |
| [fp_vs_paths_comparison.ipynb](examples/fp_vs_paths_comparison.ipynb) | Method comparison | Advanced | 20-30 min |

## 📋 Project Tracking

### [EXAMPLES_COMPLETE.md](EXAMPLES_COMPLETE.md)
Documentation of example suite completion:
- Overview of all notebooks
- New notebooks created
- Test validation
- Learning progression
- Usage statistics

## 🏗️ Project Structure

```
stochlib/
├── src/stochlib/
│   ├── __init__.py              (Main package, exports logging)
│   ├── logging_utils.py         (Enhanced logging infrastructure)
│   ├── setup.py                 (Grid, BC, IC configurations)
│   ├── boundary_conditions.py   (BC implementations)
│   ├── fokker_planck/           (FP PDE solver module)
│   ├── sde/                     (SDE path simulator module)
│   └── deterministic/           (Deterministic ODE solver)
├── tests/                       (279 comprehensive tests)
├── examples/                    (5 example notebooks)
├── README.md                    (Main overview)
├── FILE_STRUCTURE.md            (Architecture docs)
├── LOGGING_GUIDE.md            (Logging usage guide)
├── LOGGING_IMPROVEMENTS.md     (Enhancement summary)
├── EXAMPLES_COMPLETE.md        (Example suite summary)
└── pyproject.toml              (Project configuration)
```

## 🚀 Quick Start

### Installation
```bash
pip install -e .
```

### First Steps
1. Read [README.md](README.md) for overview
2. Browse [FILE_STRUCTURE.md](FILE_STRUCTURE.md) to understand organization
3. Try the first example notebook: [01_fokker_planck_intro.ipynb](examples/01_fokker_planck_intro.ipynb)
4. Follow the learning path in [examples/README.md](examples/README.md)

### Using the Logging System
```python
from stochlib import configure_logging, get_logger, log_performance
import logging

# Setup logging
configure_logging(level=logging.INFO, verbose=True, log_file="debug.log")
logger = get_logger("my_module")

# Use it
logger.info("Starting computation")

# Or decorate functions
@log_performance(logger)
def expensive_function():
    # your code here
    pass
```

### Environment Variables
```bash
# Control log level
export STOCHLIB_LOGLEVEL=DEBUG

# Specify log file
export STOCHLIB_LOG_FILE=/path/to/logfile.log

# Run your code
python your_script.py
```

## 📊 Documentation Coverage

| Area | Documentation | Examples | Tests |
|------|--------------|----------|-------|
| FP Solver | ✓ FILE_STRUCTURE | ✓ 01, 04, 05 | ✓ 279 tests |
| SDE Solver | ✓ FILE_STRUCTURE | ✓ 02, 05 | ✓ 279 tests |
| Logging | ✓ LOGGING_GUIDE | ✓ All notebooks | N/A |
| BCs | ✓ FILE_STRUCTURE | ✓ 03 | ✓ 279 tests |
| Grid Setup | ✓ FILE_STRUCTURE | ✓ All examples | ✓ 279 tests |
| Diagnostics | ✓ FILE_STRUCTURE | ✓ 01, 02, 03, 04 | ✓ 279 tests |
| Plotting | ✓ FILE_STRUCTURE | ✓ All examples | ✓ 279 tests |

## 🔗 Cross-References

### For Users Getting Started
1. Read: [README.md](README.md)
2. Learn: [examples/README.md](examples/README.md)
3. Explore: [examples/01_fokker_planck_intro.ipynb](examples/01_fokker_planck_intro.ipynb)
4. Reference: [LOGGING_GUIDE.md](LOGGING_GUIDE.md) when using logging

### For Developers
1. Study: [FILE_STRUCTURE.md](FILE_STRUCTURE.md) for architecture
2. Understand: [LOGGING_IMPROVEMENTS.md](LOGGING_IMPROVEMENTS.md) for instrumentation
3. Review: [src/stochlib/logging_utils.py](src/stochlib/logging_utils.py) for implementation
4. Check: [tests/](tests/) for comprehensive examples

### For Advanced Users
1. Advanced example: [examples/04_parameter_sweep_analysis.ipynb](examples/04_parameter_sweep_analysis.ipynb)
2. Comparison study: [examples/fp_vs_paths_comparison.ipynb](examples/fp_vs_paths_comparison.ipynb)
3. Logging patterns: [LOGGING_GUIDE.md](LOGGING_GUIDE.md#common-patterns)

## 📝 File Quick Reference

| File | Purpose | For Whom |
|------|---------|----------|
| README.md | Project overview | Everyone |
| FILE_STRUCTURE.md | Architecture & API | Developers, Advanced Users |
| LOGGING_GUIDE.md | How to use logging | Everyone |
| LOGGING_IMPROVEMENTS.md | What changed | Maintainers |
| EXAMPLES_COMPLETE.md | Example suite status | Project Managers |
| examples/README.md | How to run examples | Users |
| examples/0X_*.ipynb | Learn by doing | Users |
| pyproject.toml | Project config | Developers |
| src/stochlib/*.py | Implementation | Developers |
| tests/*.py | Validation | Developers, QA |

## ✅ Quality Metrics

- **Test Coverage**: 279 comprehensive unit tests
- **Test Status**: ✅ All passing
- **Documentation**: 5 markdown files + 5 jupyter notebooks
- **Code Quality**: Type hints, docstrings throughout
- **Examples**: 5 runnable notebooks with 80+ cells
- **Logging**: Production-ready with file support, environment control, decorators

## 🎓 Learning Paths

### Beginner (2-3 hours)
1. Read README.md (10 min)
2. Run 01_fokker_planck_intro.ipynb (15 min)
3. Read examples/README.md (10 min)
4. Run 02_sde_paths_intro.ipynb (15 min)
5. Run 03_boundary_conditions.ipynb (15 min)

### Intermediate (4-5 hours)
- Complete Beginner path
- Read FILE_STRUCTURE.md (20 min)
- Read LOGGING_GUIDE.md (15 min)
- Run 04_parameter_sweep_analysis.ipynb (20 min)
- Study source code patterns (30 min)

### Advanced (6-8 hours)
- Complete Intermediate path
- Read LOGGING_IMPROVEMENTS.md (10 min)
- Study logging_utils.py implementation (15 min)
- Run comparison notebook (30 min)
- Review test suite structure (20 min)

## 📞 Support & Reference

- **Questions about architecture?** → See [FILE_STRUCTURE.md](FILE_STRUCTURE.md)
- **How do I use logging?** → See [LOGGING_GUIDE.md](LOGGING_GUIDE.md)
- **How do I run examples?** → See [examples/README.md](examples/README.md)
- **Which example should I start with?** → Start with 01_fokker_planck_intro.ipynb
- **Are there convergence studies?** → See 04_parameter_sweep_analysis.ipynb
- **How do I test my changes?** → Run `pytest` (see README.md)

## 🔄 Documentation Update Log

| Date | Change | Document |
|------|--------|----------|
| Current | Created index | This file |
| Current | Completed example suite | EXAMPLES_COMPLETE.md |
| Current | Created 4 new notebooks | examples/0X_*.ipynb |
| Current | Enhanced logging system | LOGGING_IMPROVEMENTS.md |
| Current | Created architecture docs | FILE_STRUCTURE.md |

---

**Last Updated**: 2024
**Total Documentation**: ~3500 lines across 10 files
**Example Coverage**: 5 notebooks, 80+ cells, ~2000 lines of code
