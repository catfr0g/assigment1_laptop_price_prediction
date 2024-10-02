# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio application code
COPY . /app

# Expose the port the Gradio app runs on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]