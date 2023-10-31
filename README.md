# dbus-guesstimate

docker build --no-cache -t guesstimate . 
docker run --env-file .env -p 4500:8080 guesstimate
docker run --env-file .env -p 80:8080 997819012307.dkr.ecr.eu-central-1.amazonaws.com/guesstimate
aws configure sso

