# Docker Cheat Sheet

## Basic Commands

* `docker run -it <image_name>` - Runs an image in interactive mode with a terminal.
* `docker pull <image_name>` - Pulls an image from a registry.
* `docker exec -it <container_name> bash` - Accesses a running container's shell.
* `docker run -d <container_name>` - Runs a container in detached (background) mode.
* `docker run -it -p 8080:8080 <image_name>` - Maps port 8080 on the host to port 8080 in the container.
* `docker run -e <env_variable_name>=<variable_value> <image_name>` - Sets an environment variable.
* `docker build -t <image_name>:<tag> <path_to_docker_file>` - Builds a Docker image from a Dockerfile.
* `docker run -it --name <container_name> <image>` - Runs a container with a specific name.
* `docker inspect <container_name>` - Displays low-level information about containers, images, networks, or volumes.
* `docker network ls` - Shows all networks (default: `bridge`, `host`, `none`).
* `docker run -it --network=host <image_name>` - Runs a container using the host's network stack.
* `docker run -it --network=none <image_name>` - Runs a container with no internet access.
* `docker network create -d bridge <network_name>` - Creates a new custom bridge network.
* `docker run -it --network=<network_name> --name <container_name1> <image_name>` - Attaches a container to a specific network.
* `docker run -it -v /local/path:/container/working_dir <image_name>` - Mounts a local directory as a volume into the container.
* `docker volume create <volume_name>` - Creates a managed Docker volume.
* `docker volume ls` - Lists all Docker volumes.

---

## Docker Compose

Used to run multiple containers and manage services via a YAML file. The `version` line in `docker-compose.yml` controls which features are available.

```yaml
version: '3'
services:
  service_name:
    image: image_name
    ports:
      - "8080:8080"
    environment:
      - ENV_VARIABLE=VALUE
    restart: always  # Auto-restart if the container stops
````

**Commands:**

  * `docker-compose up -d` - Starts services in the background.
  * `docker-compose -f my-compose.yml up -d` - Starts services using a specific compose file.

-----

## Docker Networks

Docker provides default networks like `bridge`, `host`, and `none`. The `bridge` network is the default driver. Creating a custom network allows containers to communicate with each other by their service name.

```bash
docker network create -d bridge my_network
```

Containers on the same custom network can communicate directly using their container names.

-----

## Docker Volumes

Volumes are used to persist data outside the container's lifecycle. If a container is destroyed, its data remains safe in the volume.

```bash
# Bind mount a local path
docker run -it -v /local/path:/container/working_dir image_name

# Use a named volume
docker volume create my_volume
docker run -it -v my_volume:/container/data image_name
```

-----

## Dockerfile Best Practices

  * **Minimize Layers**: Each line in a Dockerfile creates a new layer. Chain commands together where possible.
  * **Avoid `latest` Tag**: Using a specific version tag (e.g., `python:3.9-slim`) makes builds more predictable.
  * **Use Minimal Base Images**: Start with small images like `alpine` or `slim` to reduce size and attack surface.
  * **Optimize Layer Caching**: Place frequently changing files (like your source code) as late as possible in the Dockerfile.
  * **Use `.dockerignore`**: Exclude unnecessary files and directories from the build context to speed up builds and reduce image size.
  * **Clean Up**: Remove build artifacts and temporary files in the same `RUN` command that created them (e.g., `apt-get clean`).

-----

## Multistage Builds

Use multistage builds to create smaller, more secure production images by separating the build environment from the runtime environment.

```dockerfile
# First stage: build the application
FROM golang:1.24 as builder
WORKDIR /app
COPY . .
RUN go build -o myapp

# Second stage: create the minimal runtime image
FROM scratch
COPY --from=builder /app/myapp /myapp
CMD ["/myapp"]
```

Build artifacts (like compiled binaries) are copied from the builder stage, leaving behind the build tools and source code.

-----

## Special Patterns

  * **Distroless Images**: Extremely minimal images containing only the application and its runtime dependencies, enhancing security by removing shells and package managers.
  * **Sidecar Pattern**: A supporting container runs alongside the main application container to provide auxiliary functions like logging, monitoring, or proxying.
  * **Adapter Pattern**: A container that standardizes and transforms output or requests to bridge communication between the main container and external systems.
  * **Ambassador Pattern**: A proxy container that simplifies network communication between the main application container and the outside world.
  * **Work Queue Pattern**: A system that uses a message queue to distribute tasks among multiple worker containers for asynchronous processing.
  * **Init Pattern**: An `init` container runs and completes a setup task (like database migration or configuration) before the main application container starts.

<!-- end list -->

```
