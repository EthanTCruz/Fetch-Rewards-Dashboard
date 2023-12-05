# Use an official Python runtime as a parent image
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

COPY ./data /usr/src/app/data


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Define environment variable
ENV dataFileLocation ./data/data_daily.csv
ENV HOST "0.0.0.0"

# Run dash_app.py when the container launches
CMD ["python", "./src/dash_app.py"]
