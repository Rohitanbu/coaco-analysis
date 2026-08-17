FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We use the CPU-only version of PyTorch to keep the image size manageable
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow scikit-learn numpy joblib scipy nptdms

# Copy model outputs and webapp
COPY thermal_classifier/outputs/best_efficientnet_b0.pt ./thermal_classifier/outputs/
COPY acoustic_classifier/outputs/best_efficientnet_b0_acoustic.pt ./acoustic_classifier/outputs/
COPY webapp/ ./webapp/
COPY templatemo_614_quantix_saas/ ./templatemo_614_quantix_saas/

# Switch to the webapp directory for execution
WORKDIR /app/webapp

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
