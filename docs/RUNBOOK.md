# AENEAS Production Runbook

## Table of Contents
1. [System Overview](#system-overview)
2. [Incident Response](#incident-response)
3. [Common Operations](#common-operations)
4. [Troubleshooting](#troubleshooting)
5. [Disaster Recovery](#disaster-recovery)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Performance Tuning](#performance-tuning)
8. [Security Procedures](#security-procedures)

---

## System Overview

### Architecture
- **Application**: FastAPI-based microservices
- **Database**: PostgreSQL 15 (Primary + Read Replicas)
- **Cache**: Redis 7.0 Cluster
- **Message Queue**: Kafka 3.6
- **Vector DB**: Qdrant 1.7.0
- **Container Orchestration**: Kubernetes (EKS)
- **Load Balancer**: AWS ALB/NLB
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack

### Critical Services
1. **aeneas-api**: Main API service (Port 8000)
2. **aeneas-worker**: Signal processing workers
3. **PostgreSQL**: Primary data store (Port 5432)
4. **Redis**: Caching layer (Port 6379)
5. **Kafka**: Event streaming (Port 9092)

---

## Incident Response

### Severity Levels
- **P1 (Critical)**: Complete service outage, data loss risk
- **P2 (High)**: Degraded service affecting >50% users
- **P3 (Medium)**: Partial service disruption
- **P4 (Low)**: Minor issues, no user impact

### Incident Response Workflow

#### 1. Detection & Triage (0-5 minutes)
```bash
# Check service health
kubectl get pods -n aeneas
kubectl get svc -n aeneas
kubectl top nodes
kubectl top pods -n aeneas

# Check recent events
kubectl get events -n aeneas --sort-by='.lastTimestamp'

# Check logs
kubectl logs -n aeneas deployment/aeneas-api --tail=100
```

#### 2. Initial Response (5-15 minutes)
```bash
# For P1/P2 incidents
# 1. Page on-call engineer
# 2. Create incident channel in Slack
# 3. Start incident timeline

# Quick health check
curl -f http://api.aeneas.io/api/v1/health

# Database connectivity
kubectl exec -n aeneas deployment/aeneas-api -- python -c "
from src.core.database import check_db_connection
check_db_connection()
"

# Redis connectivity
kubectl exec -n aeneas deployment/aeneas-api -- redis-cli -h redis-service PING
```

#### 3. Mitigation (15-30 minutes)
```bash
# Rollback deployment if needed
kubectl rollout undo deployment/aeneas-api -n aeneas

# Scale up if resource constrained
kubectl scale deployment/aeneas-api --replicas=10 -n aeneas

# Restart unhealthy pods
kubectl delete pod <pod-name> -n aeneas

# Enable circuit breaker
kubectl set env deployment/aeneas-api CIRCUIT_BREAKER_ENABLED=true -n aeneas
```

#### 4. Resolution & Post-Mortem
- Document root cause
- Update runbook with new findings
- Schedule post-mortem meeting
- Create action items for prevention

---

## Common Operations

### Deployment

#### Standard Deployment
```bash
# Build and push image
docker build -t aeneas/crypto-signals:v1.2.3 .
docker push aeneas/crypto-signals:v1.2.3

# Update Kubernetes deployment
kubectl set image deployment/aeneas-api aeneas-api=aeneas/crypto-signals:v1.2.3 -n aeneas

# Monitor rollout
kubectl rollout status deployment/aeneas-api -n aeneas

# Verify deployment
curl http://api.aeneas.io/api/v1/health
```

#### Blue-Green Deployment
```bash
# Deploy to green environment
kubectl apply -f k8s/deployment-green.yaml

# Test green environment
curl http://green.api.aeneas.io/api/v1/health

# Switch traffic
kubectl patch service aeneas-api -p '{"spec":{"selector":{"version":"green"}}}'

# Remove blue environment
kubectl delete deployment aeneas-api-blue
```

### Scaling

#### Horizontal Scaling
```bash
# Manual scaling
kubectl scale deployment/aeneas-api --replicas=5 -n aeneas

# Auto-scaling configuration
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aeneas-api-hpa
  namespace: aeneas
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aeneas-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF
```

#### Vertical Scaling
```bash
# Update resource limits
kubectl set resources deployment/aeneas-api -c=aeneas-api \
  --limits=cpu=4000m,memory=4Gi \
  --requests=cpu=2000m,memory=2Gi \
  -n aeneas
```

### Database Operations

#### Backup
```bash
# Manual backup
kubectl exec -n aeneas postgresql-0 -- pg_dump \
  -U crypto_admin -d crypto_signals \
  > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup (runs daily at 2 AM)
kubectl apply -f k8s/cronjob-backup.yaml
```

#### Restore
```bash
# Restore from backup
kubectl exec -i -n aeneas postgresql-0 -- psql \
  -U crypto_admin -d crypto_signals < backup_20240115_020000.sql

# Point-in-time recovery
kubectl exec -n aeneas postgresql-0 -- pg_basebackup \
  -D /var/lib/postgresql/data/recovery \
  -Fp -Xs -P -R \
  --target-time="2024-01-15 14:30:00"
```

#### Database Migrations
```bash
# Run migrations
kubectl exec -n aeneas deployment/aeneas-api -- alembic upgrade head

# Rollback migration
kubectl exec -n aeneas deployment/aeneas-api -- alembic downgrade -1

# Create new migration
kubectl exec -n aeneas deployment/aeneas-api -- \
  alembic revision --autogenerate -m "description"
```

### Cache Management

#### Redis Operations
```bash
# Flush cache (CAUTION: removes all cached data)
kubectl exec -n aeneas redis-master-0 -- redis-cli FLUSHALL

# Clear specific cache pattern
kubectl exec -n aeneas redis-master-0 -- redis-cli --scan --pattern "signal:*" | xargs redis-cli DEL

# Monitor cache usage
kubectl exec -n aeneas redis-master-0 -- redis-cli INFO memory

# Export cache data
kubectl exec -n aeneas redis-master-0 -- redis-cli --rdb /tmp/dump.rdb
```

---

## Troubleshooting

### High Memory Usage
```bash
# Identify memory-hungry pods
kubectl top pods -n aeneas --sort-by=memory

# Check for memory leaks
kubectl exec -n aeneas deployment/aeneas-api -- python -c "
import tracemalloc
import gc
tracemalloc.start()
# ... application code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"

# Force garbage collection
kubectl exec -n aeneas deployment/aeneas-api -- python -c "import gc; gc.collect()"

# Restart pods with memory issues
kubectl delete pod <pod-name> -n aeneas
```

### Slow API Response
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://api.aeneas.io/api/v1/health

# Enable profiling
kubectl set env deployment/aeneas-api PROFILING_ENABLED=true -n aeneas

# Check slow queries
kubectl exec -n aeneas postgresql-0 -- psql -U crypto_admin -d crypto_signals -c "
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# Analyze query plan
kubectl exec -n aeneas postgresql-0 -- psql -U crypto_admin -d crypto_signals -c "
EXPLAIN ANALYZE SELECT * FROM signals WHERE created_at > NOW() - INTERVAL '1 hour';
"
```

### Database Connection Issues
```bash
# Check connection pool
kubectl exec -n aeneas deployment/aeneas-api -- python -c "
from src.core.database import get_db_pool_stats
print(get_db_pool_stats())
"

# Increase connection pool size
kubectl set env deployment/aeneas-api DATABASE_POOL_SIZE=50 -n aeneas

# Check pgbouncer stats
kubectl exec -n aeneas pgbouncer-0 -- psql -U pgbouncer -d pgbouncer -c "SHOW POOLS;"
```

### Kafka Issues
```bash
# Check consumer lag
kubectl exec -n aeneas kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group signal-processor

# Reset consumer offset
kubectl exec -n aeneas kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group signal-processor \
  --reset-offsets --to-earliest \
  --topic crypto-signals --execute

# Check topic health
kubectl exec -n aeneas kafka-0 -- kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic crypto-signals
```

---

## Disaster Recovery

### Backup Strategy
- **Database**: Daily automated backups, 30-day retention
- **Redis**: Hourly snapshots, 7-day retention
- **Application State**: Stored in S3, versioned
- **Configuration**: GitOps with version control

### Recovery Procedures

#### Complete System Recovery
```bash
# 1. Restore infrastructure
terraform apply -auto-approve

# 2. Restore database
kubectl apply -f k8s/postgresql.yaml
kubectl exec -i -n aeneas postgresql-0 -- psql < latest_backup.sql

# 3. Restore Redis cache
kubectl apply -f k8s/redis.yaml
kubectl exec -i -n aeneas redis-master-0 -- redis-cli --rdb restore.rdb

# 4. Deploy application
kubectl apply -f k8s/

# 5. Verify system
./scripts/health_check.sh
```

#### Regional Failover
```bash
# Switch to DR region
kubectl config use-context eks-us-west-2

# Update DNS
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch file://dns-failover.json

# Deploy to DR region
kubectl apply -f k8s/

# Verify failover
curl http://dr.api.aeneas.io/api/v1/health
```

---

## Monitoring & Alerts

### Key Metrics
| Metric | Warning Threshold | Critical Threshold | Action |
|--------|------------------|-------------------|---------|
| CPU Usage | 70% | 85% | Scale horizontally |
| Memory Usage | 80% | 90% | Scale vertically |
| Response Time (p95) | 1s | 2s | Check slow queries |
| Error Rate | 1% | 5% | Check logs, rollback |
| Signal Processing Lag | 30s | 60s | Scale workers |
| Database Connections | 80% | 90% | Increase pool size |
| Cache Hit Rate | <70% | <50% | Review caching strategy |

### Alert Response

#### High CPU Alert
```bash
# Immediate actions
kubectl scale deployment/aeneas-api --replicas=+2 -n aeneas
kubectl top pods -n aeneas --sort-by=cpu

# Investigation
kubectl exec -n aeneas deployment/aeneas-api -- py-spy top --pid 1
```

#### Database Connection Pool Exhausted
```bash
# Immediate actions
kubectl set env deployment/aeneas-api DATABASE_POOL_SIZE=100 -n aeneas
kubectl rollout restart deployment/aeneas-api -n aeneas

# Long-term fix
# Review connection usage patterns
# Implement connection pooling optimization
```

---

## Performance Tuning

### Application Optimization
```python
# Enable performance features
ENVIRONMENT_VARIABLES = {
    "CACHE_ENABLED": "true",
    "CACHE_TTL": "3600",
    "REQUEST_BATCHING": "true",
    "BATCH_SIZE": "100",
    "CONNECTION_POOL_SIZE": "50",
    "ASYNC_PROCESSING": "true",
    "COMPRESSION_ENABLED": "true"
}
```

### Database Tuning
```sql
-- Update statistics
ANALYZE;

-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX CONCURRENTLY idx_signals_pair_status ON signals(pair, status);

-- Configure autovacuum
ALTER TABLE signals SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE signals SET (autovacuum_analyze_scale_factor = 0.05);
```

### Kubernetes Optimization
```yaml
# Resource optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# Node affinity for performance
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
          - m5.xlarge
          - m5.2xlarge
```

---

## Security Procedures

### Security Incident Response
1. **Isolate**: Remove affected systems from network
2. **Assess**: Determine scope and impact
3. **Contain**: Prevent further damage
4. **Eradicate**: Remove threat
5. **Recover**: Restore normal operations
6. **Review**: Post-incident analysis

### Regular Security Tasks
```bash
# Weekly vulnerability scan
./scripts/security_scan.py

# Update dependencies
pip list --outdated
pip install --upgrade <package>

# Rotate secrets
kubectl create secret generic aeneas-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# Audit logs review
kubectl logs -n aeneas deployment/aeneas-api | grep -E "AUTH|SECURITY|ERROR"
```

### Compliance Checks
- [ ] SSL certificates valid
- [ ] API rate limiting enabled
- [ ] Authentication required on all endpoints
- [ ] Sensitive data encrypted
- [ ] Audit logs retained for 90 days
- [ ] GDPR compliance verified

---

## Contact Information

### Escalation Matrix
| Level | Role | Contact | When to Contact |
|-------|------|---------|----------------|
| L1 | On-Call Engineer | PagerDuty | First response |
| L2 | Team Lead | Slack: @teamlead | P1/P2 incidents |
| L3 | Engineering Manager | Phone: +1-xxx-xxx-xxxx | Major outages |
| L4 | CTO | Email: cto@aeneas.io | Data breach/critical |

### External Contacts
- **AWS Support**: Case via AWS Console
- **Database Vendor**: support@postgresql.org
- **Security Team**: security@aeneas.io
- **Legal**: legal@aeneas.io

---

## Appendix

### Useful Commands Cheatsheet
```bash
# Quick health check
curl http://api.aeneas.io/api/v1/health | jq

# Pod logs with timestamps
kubectl logs -n aeneas deployment/aeneas-api --timestamps --tail=100

# Force pod restart
kubectl rollout restart deployment/aeneas-api -n aeneas

# Database query
kubectl exec -n aeneas postgresql-0 -- psql -U crypto_admin -d crypto_signals -c "SELECT COUNT(*) FROM signals;"

# Redis memory usage
kubectl exec -n aeneas redis-master-0 -- redis-cli MEMORY STATS

# Kafka consumer status
kubectl exec -n aeneas kafka-0 -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list
```

### Environment URLs
- **Production API**: https://api.aeneas.io
- **Staging API**: https://staging-api.aeneas.io
- **Monitoring**: https://grafana.aeneas.io
- **Logs**: https://kibana.aeneas.io
- **Traces**: https://jaeger.aeneas.io

---

*Last Updated: January 2024*
*Version: 1.0.0*
