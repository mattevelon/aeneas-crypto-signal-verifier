#!/bin/bash
# Qdrant backup script for crypto signals system

# Configuration
QDRANT_URL="http://localhost:6333"
BACKUP_DIR="./backups/qdrant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COLLECTIONS=("signals" "patterns")

# Create backup directory if it doesn't exist
mkdir -p ${BACKUP_DIR}

echo "🔄 Starting Qdrant backup..."

# Backup each collection
for COLLECTION in "${COLLECTIONS[@]}"; do
    echo "📦 Backing up collection: ${COLLECTION}"
    
    # Create snapshot
    RESPONSE=$(curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots" 2>/dev/null)
    SNAPSHOT_NAME=$(echo $RESPONSE | jq -r '.result.name')
    
    if [ "$SNAPSHOT_NAME" != "null" ] && [ -n "$SNAPSHOT_NAME" ]; then
        # Download snapshot
        curl -o "${BACKUP_DIR}/${COLLECTION}_${TIMESTAMP}.snapshot" \
            "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}" 2>/dev/null
        
        echo "✅ Collection ${COLLECTION} backed up to ${BACKUP_DIR}/${COLLECTION}_${TIMESTAMP}.snapshot"
        
        # Delete snapshot from server to save space
        curl -X DELETE "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}" 2>/dev/null
    else
        echo "❌ Failed to create snapshot for collection ${COLLECTION}"
    fi
done

# Remove old backups (keep last 7 days)
find ${BACKUP_DIR} -name "*.snapshot" -mtime +7 -delete
echo "🗑️  Old backups cleaned up"

# List recent backups
echo "📋 Recent backups:"
ls -lh ${BACKUP_DIR}/*.snapshot 2>/dev/null | tail -5

echo "✅ Qdrant backup completed!"
