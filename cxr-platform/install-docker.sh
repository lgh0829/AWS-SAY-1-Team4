#!/bin/bash

# Exit on error
set -e

# Variables
COMPOSE_VERSION="v2.24.2"
ARCH=$(uname -m)
USER_NAME=$(whoami)

echo "Updating system..."
sudo dnf update -y

echo "Installing Docker..."
sudo dnf install -y docker

echo "Enabling and starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker

echo "Adding $USER_NAME to docker group..."
sudo usermod -aG docker $USER_NAME

echo "Installing Docker Compose v${COMPOSE_VERSION}..."
if [[ "$ARCH" == "x86_64" ]]; then
  ARCH_SUFFIX="x86_64"
elif [[ "$ARCH" == "aarch64" ]]; then
  ARCH_SUFFIX="aarch64"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${ARCH_SUFFIX}" \
  -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

echo "Installation complete!"
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker-compose --version)"
echo ""
echo "Please log out and back in (or run: newgrp docker) to use Docker without sudo."