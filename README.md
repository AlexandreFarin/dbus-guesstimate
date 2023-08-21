# dbus-guesstimate

docker build --no-cache -t guesstimate . 
docker run --env-file .env -p 4500:8080 guesstimate