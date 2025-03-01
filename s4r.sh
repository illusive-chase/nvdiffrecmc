lst='air chair jugs hotdog'
for i in $lst
do
echo $i
python train.py --config configs/s4r_$i.json
done

for i in $lst
do
echo $i
python albedo_scaling.py s4r-$i
done

for i in $lst
do
echo $i
python relight.py --config configs/s4r_$i.json --envlight out/envmap6.hdr
python relight.py --config configs/s4r_$i.json --envlight out/envmap12.hdr
done

for i in $lst
do
echo $i
python relight_evaler.py s4r-$i
done
