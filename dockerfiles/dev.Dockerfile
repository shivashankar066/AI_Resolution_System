FROM 172.16.1.162:5000/ubi8/python3.9:1.0

USER root

RUN mkdir --parent /app
WORKDIR /app
COPY requirements.txt ./

RUN yum install unixODBC unixODBC-devel -y
RUN pip install pyodbc

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app

EXPOSE 8030
CMD ["python", "manage.py", "runserver", "0.0.0.0:8030"]
