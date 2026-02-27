#!/bin/bash
set -e   # Exit immediately if any command fails

echo "🚀 Starting Vault container..."
docker compose up -d vault

echo "⏳ Waiting 5 seconds for Vault to initialize..."
sleep 5

echo "🔓 Unsealing Vault..."
./vault-unseal.sh

echo "📦 Building and starting the rest of the stack..."
docker compose up -d --build 

echo "✅ All services started!"
