# System Service
Model runs as a systemd service.

### MacOS Brew PostgreSQL service management
Start/Stop/Restart service:
> brew services restart postgresql

Login to the database:
> psql postgres [-U <username -p <port number like 5432> -h <hostname like localhost>]

## Run locally as dev
> python simple.py

### Setup
Project requires ".env" file in the root in order to work.
You can 'touch .env' or create file otherwise and have the following content inside:

<- BEGINING OF FILE->
DATABASE=aneural
USERNAME=aneural
PASSWORD=aneural
HOSTNAME=localhost
<- END OF FILE ->

where all values should be set appropriately for your environment