---
trigger: always_on
---

# Cascade Rule: Manual Task Handler

## Overview
This rule governs Cascade's behavior when encountering tasks that require manual human intervention. All manual task instructions must be provided in Russian directly within the chat conversation.

## Core Principles

1. **Strict Sequential Execution**: Execute all tasks in defined order
2. **Blocking Checkpoints**: Manual tasks stop all automation until completed
3. **Russian Instructions in Chat**: Provide all instructions in Russian within the conversation
4. **Explicit Confirmation Required**: Wait for user confirmation before proceeding
5. **TASKS.md Updates Only**: Update only the TASKS.md file for task tracking

## Task Detection

### Primary Indicators
Cascade must detect manual tasks through these patterns:

- ❌ - Requires full manual intervention
- ⚠️ - Partially automated, needs manual setup  
- 🔐 - Requires credentials or authentication
- 🔧 - Requires external tool installation
- 📧 - Requires email/account verification
- 💳 - Requires payment or subscription

#### **Critical Understanding - Task Status Emojis**
**IMPORTANT**: The emojis in TASKS.md (✅, ❌, ⚠️) indicate Cascade's **ability to perform** a task, NOT completion status:
- ✅ = Cascade can perform this task automatically
- ❌ = Requires manual user intervention (API keys, external accounts, etc.)  
- ⚠️ = Partially automated or requires additional setup

### Secondary Keywords
- API credentials (api_key, api_id, api_hash, token)
- Registration/authentication (register, sign up, login)
- External services (deploy, production, domain, DNS)
- Manual indicators (manual, manually, user must, requires user)

## Behavioral Workflow

### When Manual Task Detected

1. **Stop Automation**
   - Complete all automated tasks before the manual checkpoint
   - Halt execution at manual task
   - Save current progress state

2. **Provide Russian Instructions in Chat**
   
   Structure the instructions as follows:

   ```
   🛑 **ТРЕБУЕТСЯ РУЧНОЕ ВЫПОЛНЕНИЕ**
   
   # Задача [ID]: [Название задачи]
   
   ## 📋 Краткое описание
   [Что нужно сделать]
   
   ## ⚠️ Почему требуется ручное выполнение
   [Объяснение причины]
   
   ## 🔧 Необходимые инструменты
   1. [Инструмент/Сервис] - [Ссылка]
   
   ## 📝 Пошаговые инструкции
   
   ### Шаг 1: [Действие]
   **Что делать:**
   [Подробное описание]
   
   **Команда (если есть):**
   ```bash
   [команда]
   ```
   
   **Ожидаемый результат:**
   [Что должно произойти]
   
   ### Шаг 2: [Следующее действие]
   [Продолжение инструкций]
   
   ## ✅ Проверка выполнения
   1. [Способ проверки]
   2. [Еще один способ]
   
   ## 🚨 Частые ошибки
   - **Ошибка**: [Описание]
     **Решение**: [Как исправить]
   
   ## Подтверждение
   После выполнения напишите: **done** или **выполнено**
   ```

3. **Wait for User Response**
   - Accept confirmations: `done`, `выполнено`, `готово`, `completed`
   - Error signals: `error`, `ошибка`, `проблема`, `не работает`

4. **Handle User Response**
   
   **If Confirmed:**
   - Update task status in TASKS.md:
     ```markdown
     - [x] ✅ [Task] [MANUALLY COMPLETED: DATE TIME]
     ```
   - Resume automation with next tasks
   
   **If Error Reported:**
   - Provide troubleshooting instructions in chat
   - Suggest alternative solutions
   - Wait for resolution

### Error Handling

When user reports problems, provide troubleshooting directly in chat:

```
🔧 **УСТРАНЕНИЕ ПРОБЛЕМЫ**

## Описание проблемы
[Анализ проблемы]

## Диагностика

### Проверка 1: [Что проверить]
Выполните команду:
```bash
[диагностическая команда]
```

Если результат отличается от ожидаемого:
[Инструкции по исправлению]

## Альтернативные решения
1. **Вариант А**: [Описание]
2. **Вариант Б**: [Описание]
```

## Special Cases

### API Credentials (❌)
- Provide registration steps
- Include security warnings
- Show where to store credentials

### Partial Automation (⚠️)
- Complete automated portion first
- Clearly indicate what requires manual intervention
- Provide verification steps

### Production Deployments
- Generate all necessary configurations
- Provide deployment commands
- Include rollback instructions

## Task Status Updates

Update TASKS.md immediately upon confirmation:

**Before:**
```markdown
- [ ] ❌ 2.2.1 Register Telegram application
```

**After Completion:**
```markdown
- [x] ❌ 2.2.1 Register Telegram application [MANUALLY COMPLETED: 2024-01-12 14:30]
```

## Important Guidelines

### Language Requirements
- Use simple, clear Russian
- Include English technical terms in parentheses when necessary
- Number all steps sequentially (Шаг 1, Шаг 2, etc.)

### Instruction Quality
- Be specific and detailed
- Include exact commands where applicable
- Provide expected outcomes for each step
- Address common errors proactively

### Task Sequencing
- Never skip manual tasks
- No parallel execution during manual checkpoints
- Verify completion before proceeding
- Maintain task dependency order

## Russian Language Reference

### Common Phrases
- требуется - required
- ручное выполнение - manual execution
- шаг - step
- проверка - verification
- ошибка - error
- решение - solution
- выполнено - completed
- результат - result

### Standard Messages
- "Ожидание выполнения пользователем" - Waiting for user execution
- "Задача требует ручного вмешательства" - Task requires manual intervention
- "После выполнения напишите done" - After completion write done
- "Проверьте правильность выполнения" - Verify correct execution