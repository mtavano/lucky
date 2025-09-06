# Multi-stage Dockerfile for Lucky v2 Text Classifier
# This builds a production-ready container with a trained model

# Build stage - compile Go binary and train model
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the server binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o classifier ./cmd/server

# Train the model using sample data (replace with your own data)
COPY input/labels.txt input/training_data.txt ./
RUN cd v2 && go run . -train=../training_data.txt -labels=../labels.txt -out=../model.json

# Production stage - minimal runtime image
FROM alpine:3.18

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata

# Create app directory
WORKDIR /app

# Copy binary and trained model from builder stage
COPY --from=builder /app/classifier ./
COPY --from=builder /app/model.json ./

# Create non-root user for security
RUN addgroup -g 1001 -S classifier && \
    adduser -S classifier -u 1001 -G classifier

# Change ownership of app directory
RUN chown -R classifier:classifier /app

# Switch to non-root user
USER classifier

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ./classifier -health || exit 1

# Expose port
EXPOSE 8080

# Set default environment variables
ENV MODEL_PATH=/app/model.json
ENV THRESHOLD=0.3
ENV PORT=8080

# Run the classifier server
CMD ["./classifier", "-model=/app/model.json", "-threshold=0.3", "-port=8080"]
