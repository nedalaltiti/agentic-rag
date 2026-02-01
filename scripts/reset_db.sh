#!/bin/bash
# scripts/reset_db.sh
# Resets the database by stopping containers and removing volumes

set -e

echo "🛑 Stopping containers and removing volumes (Data Wipe)..."
# -v removes named volumes declared in the `volumes` section of the Compose file
docker compose down -v

echo "🚀 Starting fresh (Migrations will run automatically)..."
docker compose up -d postgres

echo "⏳ Waiting for Postgres to initialize..."
sleep 5

echo "📋 Postgres logs:"
docker compose logs postgres
