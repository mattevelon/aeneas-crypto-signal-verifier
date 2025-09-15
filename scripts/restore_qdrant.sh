#!/bin/bash
# Qdrant restore script for crypto signals system

# Configuration
QDRANT_URL="http://localhost:6333"
BACKUP_DIR="./backups/qdrant"

# Check if backup file is provided
if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <snapshot_file>"
    echo "Available snapshots:"
    ls -lh ${BACKUP_DIR}/*.snapshot 2>/dev/null
    exit 1
fi

SNAPSHOT_FILE=$1
COLLECTION_NAME=$(basename "$SNAPSHOT_FILE" | cut -d'_' -f1)

# Check if snapshot file exists
if [ ! -f "${SNAPSHOT_FILE}" ]; then
    echo "❌ Snapshot file not found: ${SNAPSHOT_FILE}"
    exit 1
fi

echo "⚠️  WARNING: This will restore collection ${COLLECTION_NAME} from ${SNAPSHOT_FILE}"
echo "Current data in the collection will be replaced!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Restore cancelled"
    exit 0
fi

echo "🔄 Starting Qdrant restore..."

# Upload snapshot
echo "📤 Uploading snapshot..."
UPLOAD_RESPONSE=$(curl -X POST "${QDRANT_URL}/collections/${COLLECTION_NAME}/snapshots/upload" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${SNAPSHOT_FILE}" 2>/dev/null)

SNAPSHOT_NAME=$(echo $UPLOAD_RESPONSE | jq -r '.result.name')

if [ "$SNAPSHOT_NAME" != "null" ] && [ -n "$SNAPSHOT_NAME" ]; then
    # Restore from snapshot
    echo "🔄 Restoring collection from snapshot..."
    RESTORE_RESPONSE=$(curl -X PUT "${QDRANT_URL}/collections/${COLLECTION_NAME}/snapshots/${SNAPSHOT_NAME}/recover" 2>/dev/null)
    
    if echo $RESTORE_RESPONSE | grep -q "true"; then
        echo "✅ Collection ${COLLECTION_NAME} restored successfully!"
        
        # Clean up uploaded snapshot
        curl -X DELETE "${QDRANT_URL}/collections/${COLLECTION_NAME}/snapshots/${SNAPSHOT_NAME}" 2>/dev/null
    else
        echo "❌ Failed to restore collection"
        exit 1
    fi
else
    echo "❌ Failed to upload snapshot"
    exit 1
fi

# Verify collection
echo "🔍 Verifying collection..."
INFO=$(curl -s "${QDRANT_URL}/collections/${COLLECTION_NAME}")
VECTOR_COUNT=$(echo $INFO | jq -r '.result.vectors_count')
echo "📊 Collection ${COLLECTION_NAME} now has ${VECTOR_COUNT} vectors"

echo "✅ Qdrant restore completed!"
