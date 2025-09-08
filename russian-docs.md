---
description: Generate Russian documentation for tasks that cannot be completed automatically
---

# Russian Documentation Workflow for Incomplete Tasks

This workflow is triggered when encountering tasks marked with ❌ or ⚠️ that cannot be completed automatically.

## Steps

1. **Identify the incomplete task**
   - Note the task number from TASKS.md
   - Identify the reason for incompletion (credentials, manual setup, etc.)

2. **Create documentation file**
   - File name: `RUSSIAN_INSTRUCTIONS_[TASK_NUMBER].md`
   - Location: Project root directory

3. **Write documentation in Russian with this structure:**

```markdown
# Инструкции для задачи [НОМЕР]: [Описание задачи]

## Почему требуется ручное выполнение
[Объяснение причины]

## Необходимые инструменты и доступы
1. [Инструмент/Доступ 1]
2. [Инструмент/Доступ 2]

## Предварительные требования
- [Требование 1]
- [Требование 2]

## Пошаговые инструкции

### Шаг 1: [Действие]
Подробное описание действия...

### Шаг 2: [Действие]
Подробное описание действия...

## Ожидаемый результат
После выполнения всех шагов вы должны получить...

## Частые ошибки и их решения
- **Ошибка**: [Описание проблемы]
  **Решение**: [Как исправить]

## Альтернативные подходы
[Другие способы выполнения, если доступны]

## Полезные ссылки
- [Документация](URL)
- [Руководство](URL)
```

4. **Language requirements:**
   - Use simple Russian without complex technical terms
   - Include English terms in parentheses when necessary
   - Provide clear analogies for complex concepts
   - Write for beginner-level understanding

5. **Validation checklist:**
   - [ ] File created with correct naming convention
   - [ ] All sections filled with relevant content
   - [ ] Instructions are clear and numbered
   - [ ] Common errors addressed
   - [ ] Links to documentation provided
