# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the desired port
EXPOSE 9898

# Set environment variables
ENV PORT=9898
ENV HOST=0.0.0.0

# Define the command to run the bot
CMD ["sh", "./bin/start"]