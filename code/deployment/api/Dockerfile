# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY ./code/deployment/api /app

# Copy the CatBoost model into the Docker container
COPY ./models/catboost_model.cbm /app/models/catboost_model.cbm

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]