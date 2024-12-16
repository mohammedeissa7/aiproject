FROM python:3.10-slim-buster
## Step 1:
# inside container
# Create a working directory
WORKDIR /app

## Step 2:
# Outside -> inside container
# Copy source code to working directory
COPY requirements.txt requirements.txt
## Step 3:
# inside the container
# Install packages from requirements.txt
RUN pip install -r requirements.txt

COPY aiproject.py .

CMD [ "python3","aiproject"]