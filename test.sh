python lbebm3D.py --test_mode --dataset_name 3Dscene1 --model_path ./saved_models/lbebm3D_scene1.pt --device 0 --lstm_layers 1 --state_layers 3 && echo "3Dscene1 Launched." &
$P0

python lbebm3D.py --test_mode --dataset_name 3Dscene2 --model_path ./saved_models/lbebm3D_scene2.pt --device 1 --lstm_layers 1 --state_layers 3 && echo "3Dscene2 Launched." &
$P1

python lbebm3D.py --test_mode --dataset_name 3Dscene3 --model_path ./saved_models/lbebm3D_scene3.pt --device 0 --lstm_layers 1 --state_layers 3 && echo "3Dscene3 Launched." &
$P2

python lbebm3D.py --test_mode --dataset_name 3Dscene4 --model_path ./saved_models/lbebm3D_scene4.pt --device 1 --lstm_layers 1 --state_layers 3 && echo "3Dscene4 Launched." &
$P3

python lbebm3D.py --test_mode --dataset_name 3Dscene5 --model_path ./saved_models/lbebm3D_scene5.pt --device 0 --lstm_layers 1 --state_layers 3 && echo "3Dscene5 Launched." &
$P4

python lbebm3D.py --test_mode --dataset_name 3Dscene6 --model_path ./saved_models/lbebm3D_scene6.pt --device 1 --lstm_layers 1 --state_layers 3 && echo "3Dscene6 Launched." &
$P5

python lbebm3D.py --test_mode --dataset_name 3Dscene7 --model_path ./saved_models/lbebm3D_scene7.pt --device 0 --lstm_layers 1 --state_layers 3 && echo "3Dscene7 Launched." &
$P6

python lbebm3D.py --test_mode --dataset_name 3Dscene8 --model_path ./saved_models/lbebm3D_scene8.pt --device 1 --lstm_layers 1 --state_layers 3 && echo "3Dscene8 Launched." &
$P7

wait $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7