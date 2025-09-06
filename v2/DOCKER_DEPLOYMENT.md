# Lucky v2 - Docker & CI/CD Integration Guide

This guide shows how to integrate Lucky v2 into production environments using Docker containers and CI/CD pipelines.

## Overview

Lucky v2 is designed for production deployment with:
- **Containerized models**: Pre-trained models in Docker images
- **CI/CD integration**: Automated training and deployment
- **Zero-downtime updates**: Hot-swappable model files
- **Monitoring ready**: Structured logging and metrics

## Docker Examples

### Multi-stage Build with Training

```dockerfile
# Dockerfile.training
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY labels.txt training.txt ./
COPY go.mod go.sum ./
COPY . .

RUN go mod download
RUN go run ./v2 -train=training.txt -labels=labels.txt -out=model.json

FROM alpine:3.18
RUN apk --no-cache add ca-certificates
WORKDIR /app

COPY --from=builder /app/model.json ./
COPY --from=builder /app/classifier ./

RUN adduser -D -s /bin/sh classifier
USER classifier

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ./classifier -health || exit 1

EXPOSE 8080
CMD ["./classifier", "-model=model.json", "-port=8080"]
```

### External Model Loading

```dockerfile
# Dockerfile.production
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o classifier ./cmd/server

FROM alpine:3.18
RUN apk --no-cache add ca-certificates
WORKDIR /app

COPY --from=builder /app/classifier ./
RUN adduser -D -s /bin/sh classifier
USER classifier

EXPOSE 8080
CMD ["./classifier", "-port=8080"]
```

### Distroless for Security

```dockerfile
# Dockerfile.distroless
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o classifier ./cmd/server

FROM gcr.io/distroless/static-debian11

WORKDIR /app
COPY --from=builder /app/classifier ./
COPY model.json ./

EXPOSE 8080
ENTRYPOINT ["./classifier"]
CMD ["-model=model.json", "-port=8080"]
```

## HTTP Server Implementation

```go
// cmd/server/main.go
package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    v2 "github.com/mtavano/lucky/v2"
)

type Server struct {
    classifier *v2.Config
    port       string
}

type PredictionRequest struct {
    Text string `json:"text"`
}

type PredictionResponse struct {
    ID       uint    `json:"id"`
    Name     string  `json:"name"`
    Score    float64 `json:"score"`
    Duration string  `json:"duration"`
}

func main() {
    var (
        modelPath = flag.String("model", "model.json", "Path to model file")
        port      = flag.String("port", "8080", "Server port")
        threshold = flag.Float64("threshold", 0.3, "Prediction threshold")
        health    = flag.Bool("health", false, "Health check mode")
    )
    flag.Parse()

    if *health {
        resp, err := http.Get(fmt.Sprintf("http://localhost:%s/health", *port))
        if err != nil || resp.StatusCode != 200 {
            os.Exit(1)
        }
        os.Exit(0)
    }

    classifier := &v2.Config{
        Threshold: *threshold,
        Verbose:   false,
    }

    if err := classifier.LoadModel(*modelPath); err != nil {
        log.Fatalf("Failed to load model: %v", err)
    }

    server := &Server{
        classifier: classifier,
        port:       *port,
    }

    http.HandleFunc("/health", server.healthHandler)
    http.HandleFunc("/predict", server.predictHandler)
    
    log.Printf("Server starting on port %s", *port)
    log.Fatal(http.ListenAndServe(":"+*port, nil))
}

func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "status":    "healthy",
        "timestamp": time.Now().UTC().Format(time.RFC3339),
        "version":   "v2.0.0",
    })
}

func (s *Server) predictHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req PredictionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    start := time.Now()
    result := s.classifier.Predict(req.Text)
    duration := time.Since(start)

    response := PredictionResponse{
        ID:       result.ID,
        Name:     result.Name,
        Score:    result.Score,
        Duration: duration.String(),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

## CI/CD Pipeline Examples

### GitHub Actions

```yaml
# .github/workflows/model-ci.yml
name: Model CI/CD

on:
  push:
    branches: [main]
    paths: ['data/**', 'v2/**']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/lucky-classifier

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    - run: make test

  train-and-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Train model
      run: |
        cd v2
        go run . -train=../data/training.txt -labels=../data/labels.txt -out=model.json
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: Dockerfile.production
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - train
  - build
  - deploy

test:
  stage: test
  image: golang:1.21-alpine
  script:
    - make test

train-model:
  stage: train
  image: golang:1.21-alpine
  script:
    - cd v2
    - go run . -train=../data/training.txt -labels=../data/labels.txt -out=model.json
  artifacts:
    paths:
      - v2/model.json

build-image:
  stage: build
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lucky-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lucky-classifier
  template:
    metadata:
      labels:
        app: lucky-classifier
    spec:
      containers:
      - name: classifier
        image: ghcr.io/yourorg/lucky-classifier:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: lucky-classifier-service
spec:
  selector:
    app: lucky-classifier
  ports:
  - port: 80
    targetPort: 8080
```

## Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  classifier:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./model.json:/app/model.json:ro
    environment:
      - THRESHOLD=0.3
    healthcheck:
      test: ["CMD", "./classifier", "-health"]
      interval: 30s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - classifier
```

## Production Best Practices

### Security Checklist
- [ ] Use non-root user in containers
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Use HTTPS/TLS encryption
- [ ] Scan images for vulnerabilities

### Performance Optimization
- [ ] Set appropriate resource limits
- [ ] Implement connection pooling
- [ ] Add response caching
- [ ] Configure horizontal pod autoscaling

### Monitoring & Observability
- [ ] Structured logging
- [ ] Metrics collection (Prometheus)
- [ ] Health checks and probes
- [ ] Error tracking
- [ ] Performance monitoring

### Deployment Strategy
- [ ] Blue-green deployments
- [ ] Rolling updates
- [ ] Canary releases
- [ ] Rollback procedures

This guide provides the foundation for deploying Lucky v2 in production with modern containerization and CI/CD practices.
