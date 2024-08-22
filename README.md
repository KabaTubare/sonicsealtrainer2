This was setup in Docker and should be built and run from the Sonicsealtrainer2 directory using, (but that is on my M1 mac) : docker build --platform linux/amd64 -t sonicsealtraining .

docker run -e HYDRA_FULL_ERROR=1 --platform linux/amd64 sonicsealtraining
