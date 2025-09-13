---
description: Universal task completion verification against actual implementation
---

# Task Completion Verification Workflow

This workflow performs intelligent analysis of task completion claims in project tracking files against actual implementation evidence in any codebase.

## Prerequisites
- Task tracking file(s) with completion markers (checkboxes, status indicators)
- Project codebase with implementation files
- Optional: Documentation files describing manual tasks

## Workflow Steps

### Step 1: Identify and Parse Task Tracking Files
Scan for common task tracking patterns:
- Look for files: `TASKS.md`, `TODO.md`, `ROADMAP.md`, `PROJECT.md`, `tasks.txt`
- Parse task markers: `[x]`, `[ ]`, `- [x]`, `TODO:`, `DONE:`,
- Identify task hierarchies and dependencies
- Extract task IDs, descriptions, and status indicators

#### **Critical Understanding - Task Status Emojis**
**IMPORTANT**: The emojis in TASKS.md (✅, ❌, ⚠️) indicate Cascade's **ability to perform** a task, NOT completion status:
- ✅ = Cascade can perform this task automatically
- ❌ = Requires manual user intervention (API keys, external accounts, etc.)  
- ⚠️ = Partially automated or requires additional setup


### Step 2: Analyze Project Structure
Determine project type and expected artifacts:
- **Language Detection**: Check for language-specific files
  - Python: `requirements.txt`, `setup.py`, `pyproject.toml`
  - JavaScript/Node: `package.json`, `yarn.lock`, `npm-lock.json`
  - Java: `pom.xml`, `build.gradle`
  - Go: `go.mod`, `go.sum`
  - Rust: `Cargo.toml`, `Cargo.lock`
  - Ruby: `Gemfile`, `Gemfile.lock`
  - .NET: `*.csproj`, `*.sln`

- **Framework Detection**: Identify framework-specific patterns
  - Web frameworks: Django, FastAPI, Express, Spring, Rails
  - Mobile: React Native, Flutter, Swift, Kotlin
  - Desktop: Electron, Qt, GTK
  - Infrastructure: Docker, Kubernetes, Terraform

### Step 3: Verify Common Infrastructure Tasks

#### 3.1 Version Control
- Check for VCS initialization (`.git`, `.gitignore`)
- Verify commit hooks (`.husky`, `.pre-commit-config.yaml`)
- Check branch protection documentation
- Verify CI/CD configuration files

#### 3.2 Dependency Management
- Verify dependency files exist and are populated
- Check for lock files indicating resolved dependencies
- Validate development vs production dependency separation
- Check for security/vulnerability scanning setup

#### 3.3 Containerization (if applicable)
- Check for `Dockerfile` or container definitions
- Verify `docker-compose.yml` or orchestration files
- Validate container registry configurations
- Check for multi-stage builds and optimization

#### 3.4 Testing Infrastructure
- Verify test directories exist (`test/`, `tests/`, `spec/`, `__tests__/`)
- Check for test configuration files
- Validate test coverage settings
- Verify CI test execution configuration

### Step 4: Verify Database and Storage (if applicable)

#### 4.1 Database Setup
- Check for migration files/tools (Alembic, Flyway, Liquibase, ActiveRecord)
- **CRITICAL**: Verify migration directories are NOT empty
- Check for seed data or initialization scripts
- Validate connection configurations

#### 4.2 Cache and Queue Systems
- Verify cache service configurations (Redis, Memcached)
- Check message queue setup (RabbitMQ, Kafka, SQS)
- Validate persistence configurations

### Step 5: Verify API and Service Configuration

#### 5.1 API Documentation
- Check for API specification files (OpenAPI, Swagger, GraphQL schema)
- Verify API documentation generation setup
- Check for example requests/responses

#### 5.2 Configuration Management
- Verify environment variable templates (`.env.example`, `.env.template`)
- Check for configuration validation (schemas, types)
- Validate secrets management approach
- Check for environment-specific configurations

### Step 6: Verify Code Implementation

#### 6.1 Source Code Analysis
Based on project type, verify:
- Main application entry points exist
- Core modules/packages are implemented
- Not just file stubs but actual implementation
- Dependencies between modules are satisfied

#### 6.2 Code Quality Tools
- Linting configuration (ESLint, Pylint, RuboCop, etc.)
- Formatting tools (Prettier, Black, gofmt)
- Type checking (TypeScript, mypy, Flow)
- Static analysis tools

### Step 7: Cross-Reference Tasks with Implementation

For each task marked as complete:
1. **Existence Check**: Verify expected files/directories exist
2. **Content Validation**: Check files contain actual implementation, not just placeholders
3. **Integration Verification**: Confirm components work together
4. **Dependency Check**: Validate prerequisite tasks are actually complete
5. **Documentation Check**: Verify associated documentation exists

### Step 8: Detect Common False Positives

Watch for these patterns:
- Empty directories that should contain files
- Configuration files pointing to non-existent paths
- Placeholder/template values not replaced with actual values
- Services defined but not configured
- Tests written but not executable
- Documentation referencing non-existent features

### Step 9: Generate Analysis Report

Create `TASK_COMPLETION_ANALYSIS_[TIMESTAMP].md` with:

```markdown
# Task Completion Analysis Report
Generated: [ISO timestamp]
Project Type: [Detected type/framework]

## Executive Summary
- Total tasks analyzed: X
- Verified complete: X (X%)
- False positives: X
- Missing dependencies: X
- Priority fixes needed: X

## Verification Results by Category

### ✅ Fully Verified Tasks
[Tasks with complete implementation evidence]

### ⚠️ Partially Complete Tasks
[Tasks with some but not all requirements met]

### ❌ False Positive Completions
[Tasks marked complete but lacking implementation]

### 🔄 Dependency Issues
[Tasks with incomplete prerequisites]

## Evidence Summary
[File paths and specific evidence for each verification]

## Critical Gaps
[High-priority missing implementations]

## Actionable Next Steps
1. [Specific action with automation capability indicator]
2. [Commands or code needed]
3. [Manual steps required]

## Automation Opportunities
- Tasks Cascade can complete: [List]
- Tasks requiring user input: [List]
- Tasks needing external resources: [List]
```

### Step 10: Provide Recommendations

Based on findings, suggest:
- Task tracking improvements
- Verification checklist additions
- Automation opportunities
- Testing strategies
- Documentation gaps to fill

## Execution Parameters

- **Mode**: Adaptive (adjusts to project type)
- **Depth**: Deep verification with content analysis
- **Evidence**: Specific file paths and line references
- **Output**: Structured markdown report

## Usage

Trigger with: `/verify-tasks`

Optional parameters:
- `--tasks-file=<path>`: Specify custom task tracking file
- `--focus=<phase>`: Focus on specific project phase
- `--quick`: Quick verification (existence only)
- `--update`: Update task file with findings

## Universal Applicability

This workflow adapts to:
- Any programming language
- Any framework or platform
- Various task tracking formats
- Different project structures
- Multiple development methodologies

## Notes

- Performs language-agnostic verification
- Adapts checks based on detected project type
- Focuses on evidence over claims
- Identifies automation vs manual tasks
- Works with any task tracking format
