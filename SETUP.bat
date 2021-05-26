@REM docker run --add-host="localhost:192.168.1.160"

docker build -t malrl-layer3 . && docker run  ^
--add-host=local_host:192.168.1.160 ^
--add-host=local:192.168.1.160 ^
--add-host=local_host:192.168.1.160 ^
 --rm -it --network host  malrl-layer3 ^ 