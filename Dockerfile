# Use official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file if exists (you can add one)
# COPY requirements.txt .

# Install dependencies (if you have a requirements.txt)
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory contents into the container at /app
COPY . .

# Install Flask or any other dependencies directly (assuming Flask app)
RUN pip install --no-cache-dir flask

# Expose port 5000 (Flask default)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
