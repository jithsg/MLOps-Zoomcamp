# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.7-slim

RUN pip install -U pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Poetry

RUN pip install poetry

# Disable virtualenv creation by Poetry and install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Make port 9696 available to the world outside this container
EXPOSE 9696

# Run the app using Gunicorn. Replace 'app:app' with 'your_module:your_app' as needed
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
