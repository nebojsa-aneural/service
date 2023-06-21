# System Service
Model runs as a systemd service.

### MacOS Brew PostgreSQL service management
Start/Stop/Restart service:
> brew services restart postgresql

Login to the database:
> psql postgres [-U <username -p <port number like 5432> -h <hostname like localhost>]
(when developing locally, 'psql postgres' should suffice)

## Run locally as dev
> python simple.py

### Setup
Project requires ".env" file in the root in order to work.
You can 'touch .env' or create file otherwise and have the following content inside:

<- BEGINING OF FILE-><br/>
DATABASE=aneural<br/>
USERNAME=aneural<br/>
PASSWORD=aneural<br/>
HOSTNAME=localhost<br/>
<- END OF FILE -><br/>

where all values should be set appropriately for your environment

### Database setup

Install PostgreSQL on your system.
If you are on Mac or Linux use brew or apt to install it.
Once installed, execute following commands:

Create database:
> CREATE DATABASE aneural   WITH ENCODING=‘UTF8’;
Add role (crate user):
> CREATE USER aneural WITH PASSWORD 'aneural';
Grant privileges to new user on the database:
> GRANT ALL PRIVILEGES ON DATABASE aneural TO aneural;

### Docker

Build docker from Dockerfile:
> docker build -t aneural-segmentation-service .

Start bringup process:
> docker-compose up --build

Cleanup volumes and caches:
> docker system prune -a --volumes