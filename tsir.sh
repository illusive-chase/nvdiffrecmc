lst='arm ficus hotdog lego'
for i in $lst
do
echo $i
python train.py --config configs/tsir_$i.json
done

for i in $lst
do
echo $i
python albedo_scaling.py tsir-$i
done

for i in $lst
do
echo $i
python relight.py --config configs/tsir_$i.json --envlight out/bridge.hdr
python relight.py --config configs/tsir_$i.json --envlight out/city.hdr
python relight.py --config configs/tsir_$i.json --envlight out/fireplace.hdr
python relight.py --config configs/tsir_$i.json --envlight out/forest.hdr
python relight.py --config configs/tsir_$i.json --envlight out/night.hdr
done

for i in $lst
do
echo $i
python relight_evaler.py tsir-$i
done
