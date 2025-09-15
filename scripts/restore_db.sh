#!/bin/bash
# Database restore script for crypto signals system

# Configuration
DB_NAME="crypto_signals"
DB_USER="crypto_user"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="./backups"

# Check if backup file is provided
if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -lh ${BACKUP_DIR}/backup_*.sql.gz 2>/dev/null
    exit 1
fi

BACKUP_FILE=$1

# Check if backup file exists
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "❌ Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

echo "⚠️  WARNING: This will restore the database from ${BACKUP_FILE}"
echo "All current data will be lost!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Restore cancelled"
    exit 0
fi

# Create temp file for uncompressed backup
TEMP_FILE="/tmp/restore_$(date +%s).sql"

# Decompress if needed
if [[ ${BACKUP_FILE} == *.gz ]]; then
    echo "📦 Decompressing backup..."
    gunzip -c ${BACKUP_FILE} > ${TEMP_FILE}
else
    cp ${BACKUP_FILE} ${TEMP_FILE}
fi

# Drop and recreate database
echo "🔄 Dropping existing database..."
PGPASSWORD="${DB_PASSWORD:-crypto_password}" psql \
    -h ${DB_HOST} \
    -p ${DB_PORT} \
    -U ${DB_USER} \
    -d postgres \
    -c "DROP DATABASE IF EXISTS ${DB_NAME};"

echo "🔄 Creating new database..."
PGPASSWORD="${DB_PASSWORD:-crypto_password}" psql \
    -h ${DB_HOST} \
    -p ${DB_PORT} \
    -U ${DB_USER} \
    -d postgres \
    -c "CREATE DATABASE ${DB_NAME};"

# Restore the backup
echo "🔄 Restoring database..."
PGPASSWORD="${DB_PASSWORD:-crypto_password}" psql \
    -h ${DB_HOST} \
    -p ${DB_PORT} \
    -U ${DB_USER} \
    -d ${DB_NAME} \
    -f ${TEMP_FILE}

if [ $? -eq 0 ]; then
    echo "✅ Database restored successfully!"
    # Clean up temp file
    rm -f ${TEMP_FILE}
else
    echo "❌ Restore failed!"
    rm -f ${TEMP_FILE}
    exit 1
fi
