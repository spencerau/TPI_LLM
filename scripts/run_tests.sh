#!/bin/bash

clear

echo "Starting Docker services..."
docker compose down
docker compose up -d #--build

echo "Waiting for services to be healthy..."
sleep 10

echo "Running tests..."
pytest tests/ -v -s

TEST_EXIT_CODE=$?

# echo "Stopping Docker services..."
# docker compose down

exit $TEST_EXIT_CODE