
docker build -t malrl-layer3 . && docker run  ^
-p 41451:41451 ^
--add-host=local_host:192.168.1.160 ^
--add-host=local:192.168.1.160 ^
--add-host=local_host:192.168.1.160 ^
 --rm -it --network="host"  malrl-layer3 ^ z