# Use the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the container
COPY env.yml .

# Create the Conda environment
RUN conda env create -f env.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]

# Install any remaining dependencies
RUN conda install -c conda-forge -y \
    mlflow \
    pandas \
    numpy \
    scikit-learn \
    flask

# Copy
EXPOSE 5000


ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Run mlflow server when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "mle-dev","mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]