# Contributing to syft-client

Thank you for your interest in contributing to syft-client! This document provides guidelines and setup instructions for contributors.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Initial Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/OpenMined/syft-client.git
   cd syft-client
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   # Using uv (recommended)
   pip install uv
   uv pip install -e ".[test,dev]"

   # Or using pip
   pip install -e ".[test,dev]"
   ```

4. **Install pre-commit hooks**

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   This will automatically run code quality checks before each commit.

## Code Quality Standards

This project enforces code quality through automated checks. All code must pass these checks before being merged.

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit. They include:

- **black**: Code formatting (auto-fixes)
- **isort**: Import sorting (auto-fixes)
- **flake8**: Linting with complexity checks (max complexity: 10)
- **bandit**: Security vulnerability scanning
- **mypy**: Type checking
- **Standard checks**: trailing whitespace, file endings, merge conflicts, etc.

**Run manually:**

```bash
pre-commit run --all-files
```

### Linting

```bash
# Check formatting
black --check syft_client

# Auto-format code
black syft_client

# Check import order
isort --check-only syft_client

# Fix import order
isort syft_client

# Run flake8
flake8 syft_client
```

### Type Checking

```bash
mypy syft_client --no-strict-optional --ignore-missing-imports
```

### Security Scanning

```bash
bandit -r syft_client -c pyproject.toml
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style and patterns
- Keep functions focused and simple (complexity ≤ 10)
- Add type hints to new functions
- Update docstrings for modified functions

### 3. Run Quality Checks

Before committing:

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Verify no issues
black syft_client
isort syft_client
flake8 syft_client
bandit -r syft_client -c pyproject.toml
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Description of changes"
```

Pre-commit hooks will run automatically. If they fail:

- Some issues are auto-fixed (black, isort) - just commit again
- Other issues need manual fixes - fix them and commit again

**Skip hooks locally (if needed for speed):**

```bash
git commit --no-verify -m "Description of changes"
```

⚠️ **Note**: CI will still catch any skipped issues, so use this only when you're confident or need to commit quickly. The PR will fail CI checks if there are any issues.

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## CI/CD Pipeline

All PRs must pass automated checks:

1. **Lint and Format Check**: black, isort, flake8 (strict - must pass)
2. **Security Scan**: bandit (strict - must pass)
3. **Type Check**: mypy (advisory - won't block merges)
4. **Pre-commit Hooks**: All hooks must pass (strict - catches `--no-verify` skips)

These run automatically on every PR via GitHub Actions.

**Local vs CI behavior:**

- **Locally**: You can skip hooks with `--no-verify` for speed
- **In CI**: All checks run strictly and will fail the PR if there are issues
- **mypy only**: Advisory locally (manual stage), but runs in CI for visibility

## Code Style Guidelines

### Python Style

- **Line length**: 88 characters (black default)
- **Import order**: stdlib → third-party → local (isort with black profile)
- **Docstrings**: Google style
- **Type hints**: Use type hints for all public functions
- **Complexity**: Keep cyclomatic complexity ≤ 10

### Security Guidelines

⚠️ **Important for AI-Assisted Development:**

- **Never commit credentials** or API keys
- **Avoid hardcoded secrets** - use environment variables
- **Be cautious with user input** - validate and sanitize
- **Review AI-generated code** carefully for security issues
- The `bandit` security scanner will catch common issues

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

## Testing

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Integration Tests (Google Drive)

Integration tests require Google OAuth credentials to test Google Drive synchronization.

#### 1. Create credentials folder

```bash
mkdir -p credentials
```

#### 2. Get Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or select existing one)
3. Enable the **Google Drive API**:
   - Go to "APIs & Services" -> "Library"
   - Search for "Google Drive API" and click "Enable"
4. Configure **OAuth Consent Screen**:
   - Go to "APIs & Services" -> "OAuth consent screen"
   - Choose "External" (or "Internal" for Google Workspace)
   - Add scopes: `https://www.googleapis.com/auth/drive`
   - Add your test user emails
5. Add audience:
   - "APIs & Services" -> "OAuth Consent Screen" -> "Audience"
   - Add your email as a "Test users" by clicking "Add users"
6. Create **OAuth Credentials**:
   - Go to "APIs & Services" -> "Credentials"
   - Click "Create Credentials" -> "OAuth client ID"
   - Application type: **Desktop app**
   - Download the JSON files

#### 3. Generate OAuth Tokens

```bash
python scripts/create_token.py --cred-path path/to/credentials.json --token-path credentials/token_do.json
python scripts/create_token.py --cred-path path/to/credentials.json --token-path credentials/token_ds.json
```

A browser window will open for each user to authenticate.

#### 4. Set Environment Variables

```bash
export AI_AUDIT_EMAIL_DO=your_do_email@gmail.com
export AI_AUDIT_EMAIL_DS=your_ds_email@gmail.com
```

#### 5. Run Integration Tests

```bash
AI_AUDIT_EMAIL_DO=your_do_email@gmail.com AI_AUDIT_EMAIL_DS=your_ds_email@gmail.com pytest tests/integration/test_sync_manager.py -v -s
```

Or if you've already exported the variables:

```bash
pytest tests/integration/test_sync_manager.py -v -s
```

#### Security Notes

- **Never commit** credential files or tokens to git (they're in `.gitignore`)
- Tokens can be revoked at https://myaccount.google.com/permissions

## Getting Help

- **Issues**: Check existing [GitHub issues](https://github.com/OpenMined/syft-client/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/OpenMined/syft-client/discussions)
- **Questions**: Open a new issue with the `question` label

## Branch Protection Rules

The `main` branch is protected with the following rules:

- ✅ Require PR reviews (minimum 1 approval)
- ✅ Require status checks to pass:
  - Lint and Format Check
  - Security Scan
  - Type Check (advisory)
  - Pre-commit Hooks
- ✅ Require branches to be up to date before merging
- ✅ No force pushes
- ✅ No deletions

## Tips for AI-Assisted Development

When using AI coding assistants (Claude Code, GitHub Copilot, etc.):

1. **Always run pre-commit hooks** - they catch formatting and security issues
2. **Review generated code** - AI can introduce vulnerabilities
3. **Check complexity** - AI sometimes generates over-complicated code
4. **Verify type hints** - ensure they're accurate
5. **Test security** - run `bandit` on AI-generated code

The automated guardrails are specifically designed to catch common AI coding issues.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
