# Release Process Guide

This document outlines the release process for the Mutator Framework.

## 🚀 Quick Release (Automated)

### Option 1: GitHub UI (Recommended)
1. Go to your repository on GitHub
2. Click **Actions** tab
3. Click **Release** workflow on the left
4. Click **Run workflow** button
5. Select:
   - **Version bump type**: `patch`, `minor`, or `major`
   - **Prerelease**: Check if this is a prerelease
6. Click **Run workflow**

The automation will:
- ✅ Bump the version number
- ✅ Create a git tag
- ✅ Generate changelog from commits
- ✅ Create GitHub release
- ✅ Trigger PyPI publishing

### Option 2: Manual Release
If you prefer manual control:

```bash
# 1. Bump version (patch/minor/major)
bump2version patch  # or minor/major

# 2. Push changes and tags
git push origin main --tags

# 3. Create release on GitHub UI or use GitHub CLI
gh release create v$(python -c "from mutator.__version__ import __version__; print(__version__)") \
  --title "Release v$(python -c "from mutator.__version__ import __version__; print(__version__)")" \
  --generate-notes
```

## 📋 Pre-Release Checklist

Before creating a release, ensure:

- [ ] All tests pass (`pytest tests/`)
- [ ] Code is properly formatted (`black . && isort .`)
- [ ] No linting errors (`flake8 .`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if manually maintained)
- [ ] Version number follows [Semantic Versioning](https://semver.org/)

## 🔢 Version Numbering

We follow **Semantic Versioning** (SemVer):

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backward compatible

### Examples:
- Bug fix: `0.2.0` → `0.2.1`
- New feature: `0.2.1` → `0.3.0`
- Breaking change: `0.3.0` → `1.0.0`

## 🔄 Automated Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)
**Triggers**: Push to main/develop, Pull requests
- Runs tests on Python 3.8-3.12
- Code quality checks (black, isort, flake8, mypy)
- Package build verification

### 2. Release Workflow (`.github/workflows/release.yml`)
**Triggers**: Manual dispatch
- Version bumping
- Git tagging
- GitHub release creation
- Changelog generation

### 3. PyPI Publishing (`.github/workflows/publish-to-pypi.yml`)
**Triggers**: GitHub release creation
- Builds package
- Publishes to TestPyPI (always)
- Publishes to PyPI (on tagged releases)

## 🏗️ Manual Build & Test

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# Verify package
twine check dist/*

# Test installation locally
pip install dist/*.whl
python -c "import mutator; print('Success!')"

# Upload to TestPyPI (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## 🔧 PyPI Trusted Publishing Setup

For automated PyPI publishing, you need to configure **Trusted Publishing**:

### 1. PyPI Configuration
1. Go to [PyPI](https://pypi.org) → Account Settings → Publishing
2. Add a new trusted publisher:
   - **PyPI Project Name**: `mutator`
   - **Owner**: `your-github-username`
   - **Repository name**: `your-repo-name`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `pypi` (optional but recommended)

### 2. GitHub Repository Settings (Optional)
For additional security, create environments:
1. Go to Settings → Environments
2. Create environment: `pypi`
3. Add protection rules (require reviews, restrict to main branch)

## 📝 Release Notes

Release notes are automatically generated from commit messages. For better release notes:

- Use conventional commits: `feat:`, `fix:`, `docs:`, etc.
- Write descriptive commit messages
- Manually edit release notes on GitHub after creation if needed

## 🐛 Troubleshooting

### Build Fails
```bash
# Check for syntax errors
python -m py_compile mutator/**/*.py

# Check dependencies
pip install -e ".[dev]"
pytest tests/
```

### PyPI Upload Fails
- Verify package name is available
- Check version number isn't already published
- Ensure all required metadata is present
- Verify trusted publishing is configured correctly

### Version Bump Fails
```bash
# Check current version
python -c "from mutator.__version__ import __version__; print(__version__)"

# Manual version bump
bump2version --current-version 0.2.0 patch
```

## 📊 Release Metrics

After release, monitor:
- PyPI download statistics
- GitHub release downloads
- Issue reports
- Community feedback

## 🎯 Release Frequency

Recommended release schedule:
- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly or when significant features are ready
- **Major releases**: Quarterly or for breaking changes

---

## 📞 Support

For release-related issues:
1. Check GitHub Actions logs
2. Review this documentation
3. Open an issue with the `release` label 