FROM python:3
ADD src/layer1.py /
RUN pip install -r requirements.txt
CMD [ "python", "./src/layer1.py" ]