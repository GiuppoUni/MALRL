# set base image (host OS)
FROM python:3.6
# set the working directory in the container
WORKDIR /code
# copy the dependencies file to the working directory
COPY requirements.txt .
# install dependencies
RUN pip install  -Ir  requirements.txt

# copy the content of the local src directory to the working directory
COPY src .
# COPY src/inputData ./inputData

CMD [ "python", "./layer2.py", "--from-docker" ]