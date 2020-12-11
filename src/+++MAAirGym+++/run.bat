echo "Script started."
call "C:\Users\gioca\anaconda3\Scripts\activate.bat"
echo "Conda activated."
@REM start python random_main.py --random-pos --fixed-action --track-traj --can-go-back --crab-mode --draw-traj --episodes 100 &
start python random_main.py --random-pos  --track-traj --crab-mode  --episodes 100 --thickness 140 &
@REM echo "Conda activated."
@REM start python track_trajectories.py & 

