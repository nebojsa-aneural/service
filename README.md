# System Service
Model runs as a systemd service.

### MacOS Brew PostgreSQL service management
Start/Stop/Restart service:
> brew services restart postgresql

Login to the database:
> psql postgres [-U <username -p <port number like 5432> -h <hostname like localhost>]