# Инструкции для задач требующих ручного выполнения

## Задачи требующие ваших действий

### 1. Настройка Git (1.2.3, 1.2.4)

#### Включение шаблона коммитов
```bash
# Активировать шаблон коммитов глобально
git config --global commit.template ~/.gitmessage

# Или только для этого проекта
git config commit.template .gitmessage
```

#### Настройка защиты веток на GitHub
Это требует доступа к настройкам репозитория на GitHub:

1. Откройте репозиторий на GitHub
2. Перейдите в Settings → Branches
3. Нажмите "Add rule" для branch protection
4. Для ветки `main`:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (минимум 1)
   - ✅ Dismiss stale pull request approvals
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Include administrators
5. Повторите для ветки `develop` с менее строгими правилами

### 2. API Ключи и Учетные Данные (2.2.x)

#### Telegram API (2.2.1)
1. Перейдите на https://my.telegram.org
2. Войдите с вашим номером телефона
3. Выберите "API development tools"
4. Создайте новое приложение:
   - App title: "Crypto Signals Verifier"
   - Short name: "crypto_signals"
   - Platform: Other
5. Сохраните `api_id` и `api_hash`
6. Добавьте в `.env`:
   ```
   TELEGRAM_API_ID=ваш_api_id
   TELEGRAM_API_HASH=ваш_api_hash
   TELEGRAM_PHONE_NUMBER=ваш_номер
   ```

#### OpenAI API (2.2.2)
1. Зарегистрируйтесь на https://platform.openai.com
2. Перейдите в API Keys
3. Создайте новый ключ
4. Добавьте в `.env`:
   ```
   LLM_API_KEY=sk-...
   ```

#### Binance API (2.2.3)
1. Войдите в Binance
2. Перейдите в API Management
3. Создайте API с READ-ONLY правами
4. Включите IP whitelist для безопасности
5. Добавьте в `.env`:
   ```
   BINANCE_API_KEY=ваш_ключ
   BINANCE_API_SECRET=ваш_секрет
   ```

#### KuCoin API (2.2.4)
1. Войдите в KuCoin
2. Перейдите в API Management
3. Создайте API с разрешениями General и Trade (Read Only)
4. Настройте IP whitelist
5. Добавьте в `.env`:
   ```
   KUCOIN_API_KEY=ваш_ключ
   KUCOIN_API_SECRET=ваш_секрет
   KUCOIN_API_PASSPHRASE=ваша_фраза
   ```

### 3. Настройка VSCode (1.4.5)

Создайте файл `.vscode/settings.json` вручную:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--config=.flake8"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=120"],
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true
  }
}
```

Рекомендуемые расширения (`.vscode/extensions.json`):
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml",
    "redhat.vscode-yaml"
  ]
}
```

### 4. Инициализация Сервисов

#### PostgreSQL с Alembic
```bash
# Создать базу данных
docker-compose up -d postgres

# Инициализировать Alembic (если еще не сделано)
alembic init migrations

# Создать первую миграцию
alembic revision --autogenerate -m "Initial schema"

# Применить миграции
alembic upgrade head

# Создать views
docker exec -i crypto_signals_postgres psql -U crypto_user -d crypto_signals < scripts/create_views.sql
```

#### Qdrant Vector Database (3.3.2-3.3.5)
```bash
# Запустить Qdrant
docker-compose up -d qdrant

# Создать коллекцию через API
curl -X PUT 'http://localhost:6333/collections/signals' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    },
    "optimizers_config": {
      "default_segment_number": 2
    },
    "replication_factor": 1
  }'
```

### 5. Настройка Мониторинга

#### Prometheus и Grafana
```bash
# Запустить сервисы мониторинга
docker-compose up -d prometheus grafana

# Grafana доступ:
# URL: http://localhost:3000
# Login: admin
# Password: admin (измените при первом входе)

# Импортировать дашборды:
# 1. Откройте Grafana
# 2. Import → Upload JSON file
# 3. Используйте готовые дашборды для PostgreSQL, Redis, Python
```

### 6. Секретность и Безопасность

#### Настройка AWS Secrets Manager (2.1.3) - Опционально
```bash
# Установить AWS CLI
pip install awscli

# Настроить доступ
aws configure

# Создать секрет
aws secretsmanager create-secret \
  --name crypto-signals/production \
  --secret-string file://.env
```

#### Pre-commit hooks
```bash
# Установить pre-commit hooks
pre-commit install

# Проверить все файлы
pre-commit run --all-files
```

## Проверка Готовности

После выполнения всех шагов:

```bash
# 1. Проверить Docker сервисы
docker-compose ps

# 2. Проверить подключение к БД
docker exec -it crypto_signals_postgres psql -U crypto_user -d crypto_signals -c "\dt"

# 3. Проверить Redis
docker exec -it crypto_signals_redis redis-cli ping

# 4. Проверить Qdrant
curl http://localhost:6333/health

# 5. Активировать виртуальное окружение и проверить тесты
source venv/bin/activate
python -m pytest tests/test_config.py -v
```

## Частые Проблемы

**Проблема**: Permission denied для Docker
**Решение**: Добавьте пользователя в группу docker:
```bash
sudo usermod -aG docker $USER
# Перелогиньтесь
```

**Проблема**: Port already in use
**Решение**: Измените порты в docker-compose.yml или остановите конфликтующие сервисы

**Проблема**: Alembic не видит модели
**Решение**: Убедитесь что все модели импортированы в migrations/env.py

## Следующие Шаги

После завершения всех задач Phase 1:
1. Коммит изменений: `git add . && git commit -m "feat: complete Phase 1 infrastructure"`
2. Переход к Phase 2: Реализация Telegram Data Collector
