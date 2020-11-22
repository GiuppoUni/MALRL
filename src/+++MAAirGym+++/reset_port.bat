@echo off
netstat -a -n -o | findstr 41451
rem taskkill /f /PID  <PORT>