#!/bin/bash
# Database backup script for crypto signals system with point-in-time recovery

# Configuration
DB_NAME="crypto_signals"
DB_USER="crypto_user"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="./backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql"
LOG_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.log"

# Create backup directory if it doesn't exist
mkdir -p ${BACKUP_DIR}

# Perform backup
echo "🔄 Starting database backup..."
PGPASSWORD="${DB_PASSWORD:-crypto_password}" pg_dump \
    -h ${DB_HOST} \
    -p ${DB_PORT} \
    -U ${DB_USER} \
    -d ${DB_NAME} \
    -f ${BACKUP_FILE} \
    --verbose \
    --no-owner \
    --no-acl

if [ $? -eq 0 ]; then
    # Compress the backup
    gzip ${BACKUP_FILE}
    echo "✅ Backup completed: ${BACKUP_FILE}.gz"
    
    # Remove old backups (keep last 7 days)
    find ${BACKUP_DIR} -name "backup_*.sql.gz" -mtime +7 -delete
    echo "🗑️  Old backups cleaned up"
else
    echo "❌ Backup failed!"
    exit 1
fi

# List recent backups
echo "📋 Recent backups:"
ls -lh ${BACKUP_DIR}/backup_*.sql.gz | tail -5
