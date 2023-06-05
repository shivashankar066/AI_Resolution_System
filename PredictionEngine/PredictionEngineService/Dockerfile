
FROM amd64/python:3.9-buster

RUN mkdir /app
WORKDIR /app
COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y vim
RUN apt-get install -y bash-completion
RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev unixodbc-dev \
    unixodbc \
    libpq-dev
RUN apt-get install -y musl-dev

#  Install the Microsoft ODBC driver Linux.Follow the mssql documentation: https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget

#Install openssl-1.1.1

RUN wget https://www.openssl.org/source/openssl-1.1.1l.tar.gz -O openssl-1.1.1l.tar.gz && \
    tar -zxvf openssl-1.1.1l.tar.gz && cd openssl-1.1.1l && ./config && make&& \
    make install && ldconfig
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get install -y apt-transport-https
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
RUN ACCEPT_EULA=Y apt-get install -y mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
COPY . /app

EXPOSE 9000
CMD ["python", "manage.py", "runserver", "0.0.0.0:9000"]
