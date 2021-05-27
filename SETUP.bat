@REM docker run --add-host="localhost:192.168.1.160"



@REM docker-machine create --driver generic ^
@REM --generic-ip-address=192.168.0.100 ^
@REM remote-docker-host

@REM docker-machine env remote-docker-host

docker build -t malrl-layer3 . && docker run  ^
-p 41451:41451 ^
--add-host=local_host:192.168.1.160 ^
--add-host=local:192.168.1.160 ^
malrl-layer3  
@REM  --rm -it --network host ^