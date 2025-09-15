#!/bin/bash
# Comprehensive backup script for all crypto signals system components

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="./backups"
LOG_FILE="${BACKUP_ROOT}/backup_all_${TIMESTAMP}.log"

# Create backup directories
mkdir -p ${BACKUP_ROOT}/{postgres,redis,qdrant}

echo "🚀 Starting comprehensive system backup at $(date)" | tee ${LOG_FILE}

# 1. Backup PostgreSQL
echo "📊 Backing up PostgreSQL..." | tee -a ${LOG_FILE}
./scripts/backup_db.sh >> ${LOG_FILE} 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL backup completed" | tee -a ${LOG_FILE}
else
    echo "❌ PostgreSQL backup failed" | tee -a ${LOG_FILE}
fi

# 2. Backup Redis
echo "💾 Backing up Redis..." | tee -a ${LOG_FILE}
docker exec crypto_signals_redis redis-cli BGSAVE >> ${LOG_FILE} 2>&1
sleep 2
docker cp crypto_signals_redis:/data/dump.rdb ${BACKUP_ROOT}/redis/dump_${TIMESTAMP}.rdb >> ${LOG_FILE} 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Redis backup completed" | tee -a ${LOG_FILE}
else
    echo "❌ Redis backup failed" | tee -a ${LOG_FILE}
fi

# 3. Backup Qdrant
echo "🔍 Backing up Qdrant..." | tee -a ${LOG_FILE}
./scripts/backup_qdrant.sh >> ${LOG_FILE} 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Qdrant backup completed" | tee -a ${LOG_FILE}
else
    echo "❌ Qdrant backup failed" | tee -a ${LOG_FILE}
fi

# 4. Create compressed archive
echo "📦 Creating compressed archive..." | tee -a ${LOG_FILE}
tar -czf ${BACKUP_ROOT}/full_backup_${TIMESTAMP}.tar.gz \
    ${BACKUP_ROOT}/postgres/backup_*_${TIMESTAMP}.sql.gz \
    ${BACKUP_ROOT}/redis/dump_${TIMESTAMP}.rdb \
    ${BACKUP_ROOT}/qdrant/*_${TIMESTAMP}.snapshot 2>/dev/null

# 5. Clean up old backups (keep last 7 days)
echo "🗑️  Cleaning up old backups..." | tee -a ${LOG_FILE}
find ${BACKUP_ROOT} -name "full_backup_*.tar.gz" -mtime +7 -delete
find ${BACKUP_ROOT}/postgres -name "backup_*.sql.gz" -mtime +7 -delete
find ${BACKUP_ROOT}/redis -name "dump_*.rdb" -mtime +7 -delete
find ${BACKUP_ROOT}/qdrant -name "*.snapshot" -mtime +7 -delete

# 6. Report backup sizes
echo "📊 Backup Summary:" | tee -a ${LOG_FILE}
du -sh ${BACKUP_ROOT}/full_backup_${TIMESTAMP}.tar.gz 2>/dev/null | tee -a ${LOG_FILE}
echo "Total backup size:" | tee -a ${LOG_FILE}
du -sh ${BACKUP_ROOT} | tee -a ${LOG_FILE}

echo "✅ Comprehensive backup completed at $(date)" | tee -a ${LOG_FILE}
echo "📝 Log saved to: ${LOG_FILE}"
