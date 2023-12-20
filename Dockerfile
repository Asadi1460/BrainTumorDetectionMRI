# Use an official TensorFlow runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install build dependencies for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install -r requirements.txt

# Make port 8501 available to the world outside this contain 
EXPOSE 8080

# Run the Streamlit app when the container launches
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]